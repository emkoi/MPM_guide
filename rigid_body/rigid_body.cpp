// MPM_guide_elastic_solid.cpp
//
#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <execution>
#include <mutex>

constexpr size_t FPS = 60;
constexpr size_t GRID_RES = 400;
constexpr size_t SIM_RES_PX = 400;
constexpr double gridToSimFactor = static_cast<double>(SIM_RES_PX) / GRID_RES;
constexpr static double dt = 0.01;
int actualFPS = 0;

namespace Math
{
	template<typename T>
	T clamp(T val, T min, T max)
	{
		if (val < min) return min;
		if (val > max) return max;
		return val;
	}

	template<typename T>
	struct Vec2
	{
		T x;
		T y;
	};

	template<>
	struct Vec2<double>
	{
		double x;
		double y;
		double Mag() const;
	};

	double Vec2<double>::Mag() const
	{
		return std::sqrt(x * x + y * y);
	}

	template<typename T>
	struct Mat2
	{
		T m11, m12, m21, m22;
		T Det() const;
		Mat2<T> Inv() const;
		Mat2<T> Trans() const;
	};

	template<typename T>
	T Mat2<T>::Det() const
	{
		return m11 * m22 - m12 * m21;
	}

	template<typename T>
	Mat2<T> Mat2<T>::Inv() const
	{
		T det = Det();
		if (det == static_cast<T>(0.0))
			return { 0.0, 0.0, 0.0, 0.0 };
		T factor = static_cast<T>(1.0) / Det();
		return {
			factor * m22, -factor * m12,
			-factor * m21, factor * m11
		};
	}

	template<typename T>
	Mat2<T> Mat2<T>::Trans() const
	{
		return {
			m11, m21,
			m12, m22
		};
	}

	template<typename T>
	Mat2<T> operator*(T factor, const Mat2<T>& mat)
	{
		return {
			factor * mat.m11, factor * mat.m12,
			factor * mat.m21, factor * mat.m22
		};
	}

	template<typename T>
	Vec2<T> operator*(const Mat2<T>& mat, const Vec2<T>& vec)
	{
		return {
			mat.m11 * vec.x + mat.m12 * vec.y,
			mat.m21 * vec.x + mat.m22 * vec.y
		};
	}

	template<typename T>
	Mat2<T> operator*(const Mat2<T>& mat1, const Mat2<T>& mat2)
	{
		return {
			mat1.m11 * mat2.m11 + mat1.m12 * mat2.m21, mat1.m11 * mat2.m12 + mat1.m12 * mat2.m22,
			mat1.m21 * mat2.m11 + mat1.m22 * mat2.m21, mat1.m21 * mat2.m12 + mat1.m22 * mat2.m22
		};
	}

	template<typename T>
	Mat2<T> operator+(const Mat2<T>& mat1, const Mat2<T>& mat2)
	{
		return {
			mat1.m11 + mat2.m11, mat1.m12 + mat2.m12,
			mat1.m21 + mat2.m21, mat1.m22 + mat2.m22
		};
	}

	template <typename T>
	T operator*(const Vec2<T>& vec1, const Vec2<T>& vec2)
	{
		return vec1.x * vec2.x + vec1.y * vec2.y;
	}

	template <typename T>
	Vec2<T> operator*(T factor, const Vec2<T>& vec)
	{
		return { factor * vec.x, factor * vec.y };
	}

	template <typename T>
	Vec2<T> operator+(const Vec2<T>& vec1, const Vec2<T>& vec2)
	{
		return { vec1.x + vec2.x, vec1.y + vec2.y };
	}

}

using namespace Math;

Vec2<size_t> WorldCoordsToPxCoords(Vec2<double> wc)
{
	return Vec2<size_t>{(size_t)(gridToSimFactor * wc.x + 0.5), (SIM_RES_PX - (size_t)(gridToSimFactor * wc.y + 0.5))};
}

struct Particle
{
	Vec2<double> pos; // position
	Vec2<double> vel; // velocity
	double mass;
	double volume0;
	double padding; // for performance
	Mat2<double> C; // affine momentum matrix
	size_t i; // index

};

struct Cell
{
	Vec2<double> vel; // velocity
	double mass;
	double padding; // for performance
};

class Renderer
{
public:
	Renderer(sf::RenderWindow& win);
	void ShowNextFrame(const std::vector<Particle>& particles, const std::vector<Cell>& cellGrid);
	void AddParticle(const Particle& p);
private:
	static sf::Color massToColor(double cellMass);
	std::vector<sf::CircleShape> shapes;
	std::vector<sf::RectangleShape> gridRects;
	sf::RenderWindow& window;
	sf::Text fpsCounter;
	sf::Font font;
	size_t numParticles; // this is owned by the MLS_MPM_Fluid class
};

Renderer::Renderer(sf::RenderWindow& win) :
	window(win), numParticles(24025)
{
	if (!font.loadFromFile("arial.ttf"))
	{
		std::cout << "Failed to load font!" << std::endl;
	}

	fpsCounter.setFont(font);
	fpsCounter.setCharacterSize(24); // in px
	fpsCounter.setFillColor(sf::Color::White);

	gridRects.resize(GRID_RES * GRID_RES);
	shapes.resize(numParticles);
}

/// <summary>
/// needs to map [0, inf] to an RGB tuple
/// </summary>
sf::Color Renderer::massToColor(double cellMass)
{
	constexpr double minDensity = 0.0;
	constexpr double maxDensity = 0.4;
	constexpr double midDensity = (minDensity + maxDensity) / 2.0;

	cellMass = clamp(cellMass, minDensity, maxDensity);

	if (cellMass <= midDensity) {
		double dist = midDensity - cellMass;
		double pct = dist / (midDensity - minDensity);
		return sf::Color(0, 0, sf::Uint8(255 * pct + 0.5));
	}
	else {
		double dist = cellMass - midDensity;
		double pct = dist / (maxDensity - midDensity);
		return sf::Color(0, sf::Uint8(127 * pct + 0.5), 0);
	}
}

void Renderer::ShowNextFrame(const std::vector<Particle>& particles, const std::vector<Cell>& cellGrid)
{
	window.clear(sf::Color(0, 0, 0));

	//size_t gridInd = 0;
	//constexpr float cellSquareSize = static_cast<float>(gridToSimFactor) + 1;
	//sf::RectangleShape rect({ cellSquareSize, cellSquareSize });
	//for (size_t gridY = 0; gridY < GRID_RES; ++gridY) {
	//	for (size_t gridX = 0; gridX < GRID_RES; ++gridX, ++gridInd) {
	//		sf::Color fillColor = massToColor(cellGrid[gridInd].mass);
	//		rect.setFillColor(fillColor);
	//		Vec2<size_t> pxCoords = WorldCoordsToPxCoords({ (double)gridX, (double)gridY + 1 });
	//		rect.setPosition((float)pxCoords.x, (float)pxCoords.y);
	//		gridRects[gridInd] = rect;
	//	}
	//}

	//for (auto it = particles.begin(); it < particles.end(); ++it)
	//	AddParticle(*it);

	sf::CircleShape circle(0.6f, 6);
	circle.setFillColor(sf::Color(230, 230, 230));
	for (size_t iPart = 0; iPart < numParticles; ++iPart) {
		Vec2<size_t> pxCoords = WorldCoordsToPxCoords(particles[iPart].pos);
		circle.setPosition((float)pxCoords.x, (float)pxCoords.y);
		shapes[iPart] = circle;
	}

	for (sf::Shape& shape : gridRects)
		window.draw(shape);
	//gridRects.clear();

	for (sf::Shape& shape : shapes)
		window.draw(shape);
	//shapes.clear();

	fpsCounter.setString(sf::String(std::string("FPS: ") + std::to_string(actualFPS)));
	window.draw(fpsCounter);

	window.display();
}

void Renderer::AddParticle(const Particle& p)
{
	sf::CircleShape circle(0.6f, 6);
	circle.setFillColor(sf::Color(230, 230, 230));
	Vec2<size_t> pxCoords = WorldCoordsToPxCoords(p.pos);
	circle.setPosition((float)pxCoords.x, (float)pxCoords.y);
	shapes.push_back(circle);
}

double uRandRange(double start, double end)
{
	static std::random_device rd;  //Will be used to obtain a seed for the random number engine
	static std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	static std::uniform_real_distribution<> urand(0.0, 1.0);
	return start + (end - start) * urand(gen);
}

class World
{
public:
	std::vector<Cell> mCellGrid;

};

/// <summary>
/// For rigid solids, probably can just implement a square with collision detection against every particle in nearby cells.
/// For every particle, if a collision with a rigid object occurs, we can move the particle along a normal vector to the surface of the rigid object and reflect its momentum.
/// Then, we can apply 
/// </summary>
class MLS_MPM_Solid
{
public:
	MLS_MPM_Solid(sf::RenderWindow& window);
	virtual ~MLS_MPM_Solid();
	void Start();
	void Update();
	void HandleMouseInteraction();
	void Simulate();
	void SpawnBox(int x, int y, int boxX = 8, int boxY = 8);
	void JobP2G();
	void JobClearGrid();
	void JobUpdateGrid();
	void JobG2P();

	constexpr static size_t gridRes = GRID_RES;
	constexpr static size_t numCells = gridRes * gridRes;

	constexpr static int iterations = static_cast<int>(1.0 / dt);

	constexpr static double gravity = -0.3;

	// Lamé parameters for stress-strain relationship
	constexpr static double elasticLambda = 0.1; // too high = bugs
	constexpr static double elasticMu = 1000.0;

	size_t mNumParticles;
	std::vector<Particle> mParticleArray;
	std::vector<Cell> mCellGrid;
	Mat2<double>* mDeformationGradient;
	std::vector<Vec2<double>> mTempPositions;

	// quadratic interpolation mWeights
	Vec2<double> mWeights[3];

	sf::RenderWindow& window;
	Renderer renderer;

	const double mouseRadius = 10.0;
	bool mouseDown = false;
	Vec2<double> mousePos{ 0.0, 0.0 };
};

MLS_MPM_Solid::MLS_MPM_Solid(sf::RenderWindow& window) :
	window(window),
	renderer(window),
	mDeformationGradient(nullptr),
	mNumParticles(0),
	mWeights{ 0, 0 }
{

}

MLS_MPM_Solid::~MLS_MPM_Solid()
{
	if (mDeformationGradient)
		delete[] mDeformationGradient;
}

void MLS_MPM_Solid::Start()
{
	mTempPositions.clear();
	SpawnBox(gridRes / 2, gridRes / 2, 32, 32);
	mNumParticles = mTempPositions.size();

	mParticleArray.resize(mNumParticles);
	mDeformationGradient = new Mat2<double>[mNumParticles];

	// populate our array of particles, set their initial state
	const Vec2<double> vel{ 0.0, 0.0 };
	const double mass = 1.0;
	const double volume0 = 0.0;
	const double padding = 0.0;
	const Mat2<double> identityMat{
		1.0, 0.0,
		0.0, 1.0
	};
	const Mat2<double> C{ identityMat };
	const Mat2<double> deformationGrad{ identityMat };
	for (size_t i = 0; i < mNumParticles; ++i)
	{
		Particle p{
			mTempPositions[i],
			vel,
			mass,
			volume0,
			padding,
			C,
			i
		};
		mParticleArray[i] = p;

		mDeformationGradient[i] = deformationGrad;
	}

	mCellGrid.resize(numCells);
	for (size_t i = 0; i < numCells; ++i)
	{
		mCellGrid[i].mass = 0.0;
		mCellGrid[i].padding = 0.0;
		mCellGrid[i].vel = { 0.0, 0.0 };
	}

	// ---- begin precomputation of particle volumes
	// MPM course, equation 152 

	// launch a P2G job to scatter particle mass to the grid
	JobP2G(); // <-- parallelize this

	for (auto it = mParticleArray.begin(); it < mParticleArray.end(); ++it)
	{
		Particle& p = *it;

		// quadratic interpolation mWeights
		Vec2<double> cellIdx{ std::floor(p.pos.x), std::floor(p.pos.y) };
		Vec2<double> cellDiff{
			p.pos.x - cellIdx.x - 0.5,
			p.pos.y - cellIdx.y - 0.5
		};
		mWeights[0] = { 0.5 * std::pow(0.5 - cellDiff.x, 2.0), 0.5 * std::pow(0.5 - cellDiff.y, 2.0) };
		mWeights[1] = { 0.75 - std::pow(cellDiff.x, 2.0), 0.75 - std::pow(cellDiff.y, 2.0) };
		mWeights[2] = { 0.5 * std::pow(0.5 + cellDiff.x, 2.0), 0.5 * std::pow(0.5 + cellDiff.y, 2.0) };

		double density = 0.0;
		// iterate over neighbouring 3x3 cells
		for (size_t gx = 0; gx < 3; ++gx)
		{
			for (size_t gy = 0; gy < 3; ++gy)
			{
				double weight = mWeights[gx].x * mWeights[gy].y;

				// map 2D to 1D index in grid
				size_t cellIndex = ((size_t)cellIdx.x + gx - 1) * gridRes + ((size_t)cellIdx.y + gy - 1);
				density += mCellGrid[cellIndex].mass * weight;
			}
		}

		// per-particle volume estimate has now been computed
		double volume = p.mass / density;
		p.volume0 = volume;
	}
}

void MLS_MPM_Solid::Update()
{
	HandleMouseInteraction();

	for (int i = 0; i < iterations; ++i)
	{
		Simulate();
	}

	renderer.ShowNextFrame(mParticleArray, mCellGrid);
}

void MLS_MPM_Solid::HandleMouseInteraction()
{
	return;
}

void MLS_MPM_Solid::Simulate()
{
	//Profiler.BeginSample("ClearGrid");
	//new Job_ClearGrid(){
	//	grid = grid
	//}.Schedule(num_cells, division).Complete();
	//Profiler.EndSample();
	JobClearGrid();

	// P2G, first round
	//Profiler.BeginSample("P2G");
	//new Job_P2G(){
	//	ps = ps,
	//	Fs = Fs,
	//	grid = grid,
	//	num_particles = num_particles
	//}.Schedule().Complete();
	//Profiler.EndSample();
	JobP2G();

	//Profiler.BeginSample("Update grid");
	//new Job_UpdateGrid(){
	//	grid = grid
	//}.Schedule(num_cells, division).Complete();
	//Profiler.EndSample();
	JobUpdateGrid();

	//Profiler.BeginSample("G2P");
	//new Job_G2P(){
	//	ps = ps,
	//	Fs = Fs,
	//	mouse_down = mouse_down,
	//	mouse_pos = mouse_pos,
	//	grid = grid
	//}.Schedule(num_particles, division).Complete();
	//Profiler.EndSample();
	JobG2P();

}

void MLS_MPM_Solid::SpawnBox(int x, int y, int boxX, int boxY)
{
	const double spacing = 0.5;
	for (double i = -boxX / 2; i < boxX / 2; i += spacing)
	{
		for (double j = -boxY / 2; j < boxY / 2; j += spacing)
		{
			Vec2<double> pos{ x + i, y + j };
			mTempPositions.push_back(pos);
		}
	}
}


/// <summary>
/// - implements constitutive model
/// - rigid objects likely wont need a grid
/// </summary>
void MLS_MPM_Solid::JobP2G()
{
	////std::vector<Cell> cellGridCpy = mCellGrid;
	//static auto jobP2G = [this](Particle& p) -> void
	//{
	for (auto it = mParticleArray.begin(); it < mParticleArray.end(); ++it)
	{
		Particle& p = *it;
		// deformation gradient (jacobians)
		// characterizes expansion, contraction, shearing, etc. But also, notably, rotations which involve no local distortion.
		Mat2<double> defGrad = mDeformationGradient[p.i];
		volatile Mat2<double> defGrad2 = mDeformationGradient[p.i];

		// jacobian determinant
		// should always be 1 for a nondeformable object
		//double J = defGrad.Det();
		double J = 1.0;

		// MPM course, page 46
		double volume = p.volume0 * J;

		// useful matrices for Neo-Hookean model
		Mat2<double> defGradTrans{ defGrad.Trans() }; // transpose of a rotation matrix = inverse
		Mat2<double> defGradTransInv{ defGradTrans.Inv() }; // this equals defGrad
		//var F_minus_F_inv_T = F - F_inv_T;
		Mat2<double> defGradSubDefGradTransInv{ // this whole thing is a 0 matrix
			defGrad.m11 - defGradTransInv.m11, defGrad.m12 - defGradTransInv.m12,
			defGrad.m21 - defGradTransInv.m21, defGrad.m22 - defGradTransInv.m22
		};

		// MPM course equation 48
		Mat2<double> Pterm0 = elasticMu * defGradSubDefGradTransInv; /// defGradSubDefGradTransInv is 0, but elasticMu is infinity. This is probably 0.
		Mat2<double> Pterm1 = elasticLambda * std::log(J) * defGradTransInv; /// J = 1, so this = 0
		Mat2<double> P{ // 0 matrix
			Pterm0.m11 + Pterm1.m11, Pterm0.m12 + Pterm1.m12,
			Pterm0.m21 + Pterm1.m21, Pterm0.m22 + Pterm1.m22
		};

		// cauchy_stress = (1 / det(F)) * P * F_T
		// equation 38, MPM course
		//stress = (1.0f / J) * math.mul(P, F_T);
		Mat2<double> stress = 1.0 / J * P * defGradTrans; // seems like 0 again as P is 0

		// (M_p)^-1 = 4, see APIC paper and MPM course page 42
		// this term is used in MLS-MPM paper eq. 16. with quadratic mWeights, Mp = (1/4) * (delta_x)^2.
		// in this simulation, delta_x = 1, because i scale the rendering of the domain rather than the domain itself.
		// we multiply by dt as part of the process of fusing the momentum and force update for MLS-MPM
		//var eq_16_term_0 = -volume * 4 * stress * dt;
		Mat2<double> eq16Term0 = -volume * 4 * dt * stress; // again, 0

		// quadratic interpolation mWeights
		//uint2 cell_idx = (uint2)p.x;
		Vec2<size_t> cellIdx{ (size_t)p.pos.x, (size_t)p.pos.y };
		//float2 cell_diff = (p.x - cell_idx) - 0.5f;
		Vec2<double> cellDiff{ p.pos.x - cellIdx.x - 0.5, p.pos.y - cellIdx.y - 0.5 };
		Vec2<double> weights[3];
		//mWeights[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
		weights[0] = { 0.5 * (0.5 - cellDiff.x) * (0.5 - cellDiff.x), 0.5 * (0.5 - cellDiff.y) * (0.5 - cellDiff.y) };
		//mWeights[1] = 0.75f - math.pow(cell_diff, 2);
		weights[1] = { 0.75 - cellDiff.x * cellDiff.x, 0.75 - cellDiff.y * cellDiff.y };
		//mWeights[2] = 0.5f * math.pow(0.5f + cell_diff, 2);
		weights[2] = { 0.5 * (0.5 + cellDiff.x) * (0.5 + cellDiff.x), 0.5 * (0.5 + cellDiff.y) * (0.5 + cellDiff.y) };

		// for all surrounding 9 cells
		for (size_t gx = 0; gx < 3; ++gx)
		{
			for (size_t gy = 0; gy < 3; ++gy)
			{
				double weight = weights[gx].x * weights[gy].y;

				//uint2 cell_x = math.uint2(cell_idx.x + gx - 1, cell_idx.y + gy - 1);
				Vec2<size_t> cellCoords{ cellIdx.x + gx - 1, cellIdx.y + gy - 1 };
				//float2 cell_dist = (cell_x - p.x) + 0.5f;
				Vec2<double> cellDist{ cellCoords.x - p.pos.x + 0.5, cellCoords.y - p.pos.y + 0.5 };
				//float2 Q = math.mul(p.C, cell_dist);
				Vec2<double> Q{ p.C.m11 * cellDist.x + p.C.m12 * cellDist.y, p.C.m21 * cellDist.x + p.C.m22 * cellDist.y };

				// scatter mass and momentum to the grid
				//int cell_index = (int)cell_x.x * grid_res + (int)cell_x.y;
				size_t cellIdx = cellCoords.x * gridRes + cellCoords.y;
				//Cell cell = grid[cell_index];
				Cell& cell = mCellGrid[cellIdx];

				// MPM course, equation 172
				double weightedMass = weight * p.mass;
				cell.mass += weightedMass;

				// APIC P2G momentum contribution
				// dpX: lowercase p = MOMENTUM
				const double dpX = weightedMass * (p.vel.x + Q.x);
				const double dpY = weightedMass * (p.vel.y + Q.y);
				const Vec2<double> dp{ dpX, dpY };
				cell.vel.x += dp.x;
				cell.vel.y += dp.y;

				// fused force/momentum update from MLS-MPM
				// see MLS-MPM paper, equation listed after eqn. 28
				//float2 momentum = math.mul(eq_16_term_0 * weight, cell_dist);
				Vec2<double> momentum = weight * eq16Term0 * cellDist; // eq16Term0 = 0... so 0! (clearly wrong) :/
				cell.vel.x += momentum.x;
				cell.vel.y += momentum.y;

				// total update on cell.v is now:
				// weight * (dt * M^-1 * p.volume * p.stress + p.mass * p.C)
				// this is the fused momentum + force from MLS-MPM. however, instead of our stress being derived from the energy density,
				// i use the weak form with cauchy stress. converted:
				// p.volume_0 * (dΨ/dF)(Fp)*(Fp_transposed)
				// is equal to p.volume * σ

				// note: currently "cell.v" refers to MOMENTUM, not velocity!
				// this gets converted in the UpdateGrid step below...
			}
		}
	}
	//};

	//std::for_each(std::execution::seq, mParticleArray.begin(), mParticleArray.end(), jobP2G);
}

void MLS_MPM_Solid::JobClearGrid()
{
	for (size_t i = 0; i < numCells; ++i)
	{
		Cell& cell = mCellGrid[i];

		// reset grid scratch-pad entirely
		cell.mass = 0;
		cell.vel = { 0.0, 0.0 };
	}
}

void MLS_MPM_Solid::JobUpdateGrid()
{
	for (size_t i = 0; i < numCells; ++i)
	{
		Cell& cell = mCellGrid[i];

		if (cell.mass > 0)
		{
			// convert momentum to velocity, apply gravity
			cell.vel.x /= cell.mass;
			cell.vel.y /= cell.mass;

			const double dvY = dt * gravity;
			cell.vel.y += dvY; // classical
			//cell.vel.y = (cell.vel.y + dvY) / (1 + cell.vel.y * c / (c * c)); // relativistic



			// 'slip' boundary conditions
			size_t x = i / gridRes;
			size_t y = i % gridRes;
			if (x < 2 || x > gridRes - 3)
				cell.vel.x = 0.0;
			if (y < 2 || y > gridRes - 3)
				cell.vel.y = 0.0;
		}
	}
}

void MLS_MPM_Solid::JobG2P()
{
	static auto jobG2P = [this](Particle& p) -> void {
		// reset particle velocity. we calculate it from scratch each step using the grid
		p.vel = { 0.0, 0.0 };

		// quadratic interpolation mWeights
		Vec2<size_t> cellIdx{ (size_t)p.pos.x, (size_t)p.pos.y };
		Vec2<double> cellDiff{ p.pos.x - cellIdx.x - 0.5, p.pos.y - cellIdx.y - 0.5 };
		Vec2<double> weights[3];
		weights[0] = { 0.5 * (0.5 - cellDiff.x) * (0.5 - cellDiff.x), 0.5 * (0.5 - cellDiff.y) * (0.5 - cellDiff.y) };
		weights[1] = { 0.75 - cellDiff.x * cellDiff.x, 0.75 - cellDiff.y * cellDiff.y };
		weights[2] = { 0.5 * (0.5 + cellDiff.x) * (0.5 + cellDiff.x), 0.5 * (0.5 + cellDiff.y) * (0.5 + cellDiff.y) };

		// constructing affine per-particle momentum matrix from APIC / MLS-MPM.
		// see APIC paper (https://web.archive.org/web/20190427165435/https://www.math.ucla.edu/~jteran/papers/JSSTS15.pdf), page 6
		// below equation 11 for clarification. this is calculating C = B * (D^-1) for APIC equation 8,
		// where B is calculated in the inner loop at (D^-1) = 4 is a constant when using quadratic interpolation functions
		Mat2<double> B{ 0.0, 0.0, 0.0, 0.0 };
		for (size_t gx = 0; gx < 3; ++gx)
		{
			for (size_t gy = 0; gy < 3; ++gy)
			{
				double weight = weights[gx].x * weights[gy].y;

				Vec2<size_t>cellX{ cellIdx.x + gx - 1, cellIdx.y + gy - 1 };
				size_t cellIndex = cellX.x * gridRes + cellX.y;

				Vec2<double> dist{
					cellX.x - p.pos.x + 0.5,
					cellX.y - p.pos.y + 0.5
				};
				Vec2<double> weightedVelocity{
					mCellGrid[cellIndex].vel.x * weight,
					mCellGrid[cellIndex].vel.y * weight
				};

				// APIC paper equation 10, constructing inner term for B
				//var term = math.float2x2(weighted_velocity * dist.x, weighted_velocity * dist.y);
				Mat2<double> term{
					weightedVelocity.x * dist.x, weightedVelocity.x * dist.y,
					weightedVelocity.y * dist.x, weightedVelocity.y * dist.y
				};

				B.m11 += term.m11;
				B.m12 += term.m12;
				B.m21 += term.m21;
				B.m22 += term.m22;

				p.vel.x += weightedVelocity.x;
				p.vel.y += weightedVelocity.y;
			}
		}
		p.C = 4.0 * B;

		// advect particles
		p.pos.x += p.vel.x * dt;
		p.pos.y += p.vel.y * dt;

		// safety clamp to ensure particles don't exit simulation domain
		p.pos.x = std::max(1.0, std::min((double)gridRes - 2, p.pos.x));
		p.pos.y = std::max(1.0, std::min((double)gridRes - 2, p.pos.y));

		//// mouse interaction
		//if (mouse_down) {
		//	var dist = p.x - mouse_pos;
		//	if (math.dot(dist, dist) < mouse_radius * mouse_radius) {
		//		float norm_factor = (math.length(dist) / mouse_radius);
		//		norm_factor = math.pow(math.sqrt(norm_factor), 8);
		//		var force = math.normalize(dist) * norm_factor * 0.5f;
		//		p.v += force;
		//	}
		//}

		// deformation gradient update - MPM course, equation 181
		// Fp' = (I + dt * p.C) * Fp
		Mat2<double> defGradNew{
			1.0, 0.0,
			0.0, 1.0
		};

		defGradNew.m11 += dt * p.C.m11;
		defGradNew.m12 += dt * p.C.m12;
		defGradNew.m21 += dt * p.C.m21;
		defGradNew.m22 += dt * p.C.m22;

		mDeformationGradient[p.i] = defGradNew * mDeformationGradient[p.i];
	};

	std::for_each(std::execution::par_unseq, mParticleArray.begin(), mParticleArray.end(), jobG2P);
}




class MLS_MPM_Fluid
{
public:
	MLS_MPM_Fluid(sf::RenderWindow& window);
	virtual ~MLS_MPM_Fluid();
	void Start();
	void Update();
	void HandleMouseInteraction();
	void Simulate();
	void SpawnBox(int x, int y, int boxX = 8, int boxY = 8);
	void JobP2G1();
	void JobP2G2();
	void JobClearGrid();
	void JobUpdateGrid();
	void JobG2P();

	constexpr static size_t gridRes = GRID_RES;
	constexpr static size_t numCells = gridRes * gridRes;

	constexpr static int iterations = static_cast<int>(1.0 / dt);

	constexpr static double gravity = -0.3;

	// fluid parameters
	constexpr static double restDensity = 0.6;
	constexpr static double dynamicViscosity = 0.1;
	// equation of state
	constexpr static double eosStiffness = 20.0; // compressibility
	constexpr static double eosPower = 4.0;

	size_t numParticles;
	std::vector<Particle> particleArray;
	std::vector<Cell> cellGrid;
	std::vector<Vec2<double>> tempPositions;

	// quadratic interpolation weights
	Vec2<double> weights[3];

	sf::RenderWindow& window;
	Renderer renderer;

	const double mouseRadius = 10.0;
	bool mouseDown = false;
	Vec2<double> mousePos{ 0.0, 0.0 };

private:
	std::vector<std::tuple<bool, size_t, Cell>>P2G1CellGridWrites;
}; 

MLS_MPM_Fluid::MLS_MPM_Fluid(sf::RenderWindow& window) :
	window(window),
	renderer(window),
	numParticles(0),
	weights{ 0, 0 }
{

}

MLS_MPM_Fluid::~MLS_MPM_Fluid()
{

}

void MLS_MPM_Fluid::Start()
{
	tempPositions.clear();
	SpawnBox((int)(15 / 32.0 * gridRes), (int)(5 / 8.0 * gridRes), gridRes / 2, gridRes / 2);
	numParticles = tempPositions.size();

	particleArray.resize(numParticles);

	P2G1CellGridWrites.resize(numParticles * 9);
	P2G1CellGridWrites.assign(numParticles * 9, { false, 0, Cell() });

	// populate our array of particles, set their initial state
	const Vec2<double> vel{ 0.0, 0.0 };
	const double mass = 1.0;
	const double volume0 = 0.0;
	const double padding = 0.0;
	const Mat2<double> identityMat{
		1.0, 0.0,
		0.0, 1.0
	};
	const Mat2<double> C{ identityMat };
	const Mat2<double> deformationGrad{ identityMat };
	for (size_t i = 0; i < numParticles; ++i)
	{
		Particle p{
			tempPositions[i],
			vel,
			mass,
			volume0,
			padding,
			C,
			i
		};
		particleArray[i] = p;
	}

	cellGrid.resize(numCells);
	for (size_t i = 0; i < numCells; ++i)
	{
		cellGrid[i].mass = 0.0;
		cellGrid[i].padding = 0.0;
		cellGrid[i].vel = { 0.0, 0.0 };
	}
}

void MLS_MPM_Fluid::Update()
{
	HandleMouseInteraction();

	for (int i = 0; i < iterations; ++i)
	{
		Simulate();
	}

	renderer.ShowNextFrame(particleArray, cellGrid);
}

void MLS_MPM_Fluid::HandleMouseInteraction()
{
	return;
}

void MLS_MPM_Fluid::Simulate()
{
	//Profiler.BeginSample("ClearGrid");
	//new Job_ClearGrid(){
	//	grid = grid
	//}.Schedule(num_cells, division).Complete();
	//Profiler.EndSample();
	JobClearGrid();

	//// P2G, first round
	//Profiler.BeginSample("P2G 1");
	//new Job_P2G_1(){
	//	ps = ps,
	//	grid = grid,
	//	num_particles = num_particles
	//}.Schedule().Complete();
	//Profiler.EndSample();
	JobP2G1();

	//// P2G, second round      
	//Profiler.BeginSample("P2G 2");
	//new Job_P2G_2(){
	//	ps = ps,
	//	grid = grid,
	//	num_particles = num_particles
	//}.Schedule().Complete();
	//Profiler.EndSample();
	JobP2G2();

	//Profiler.BeginSample("Update grid");
	//new Job_UpdateGrid(){
	//	grid = grid
	//}.Schedule(num_cells, division).Complete();
	//Profiler.EndSample();
	JobUpdateGrid();

	//Profiler.BeginSample("G2P");
	//new Job_G2P(){
	//	ps = ps,
	//	Fs = Fs,
	//	mouse_down = mouse_down,
	//	mouse_pos = mouse_pos,
	//	grid = grid
	//}.Schedule(num_particles, division).Complete();
	//Profiler.EndSample();
	JobG2P();

}

void MLS_MPM_Fluid::SpawnBox(int x, int y, int boxX, int boxY)
{
	// particles automatically spaced at their rest density
	const double spacing = sqrt(1.0 / restDensity);

	for (double i = -boxX / 2; i < boxX / 2; i += spacing)
	{
		for (double j = -boxY / 2; j < boxY / 2; j += spacing)
		{
			Vec2<double> pos{ x + i, y + j };
			tempPositions.push_back(pos);
		}
	}
}

std::mutex m;
/// <summary>
/// implements constitutive model
/// </summary>
void MLS_MPM_Fluid::JobP2G1()
{
	////std::vector<Cell> cellGridCpy = cellGrid;
	static auto jobP2G = [this](Particle& p) -> void
	{
	size_t particleIdx = 0;
	//for (auto it = particleArray.begin(); it < particleArray.end(); ++it)
	//{
	//	Particle& p = *it;
		// counts 0, 1, 2...
		particleIdx = (&p - &*particleArray.begin());

		Vec2<size_t> cellCoords{ (size_t)p.pos.x, (size_t)p.pos.y };
		Vec2<double> cellDiff{ p.pos.x - cellCoords.x - 0.5, p.pos.y - cellCoords.y - 0.5 };

		Vec2<double> weights[3];
		//weights[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
		weights[0] = { 0.5 * (0.5 - cellDiff.x) * (0.5 - cellDiff.x), 0.5 * (0.5 - cellDiff.y) * (0.5 - cellDiff.y) };
		//weights[1] = 0.75f - math.pow(cell_diff, 2);
		weights[1] = { 0.75 - cellDiff.x * cellDiff.x, 0.75 - cellDiff.y * cellDiff.y };
		//weights[2] = 0.5f * math.pow(0.5f + cell_diff, 2);
		weights[2] = { 0.5 * (0.5 + cellDiff.x) * (0.5 + cellDiff.x), 0.5 * (0.5 + cellDiff.y) * (0.5 + cellDiff.y) };

		Mat2<double> C = p.C;

		size_t neighborCount = 0;
		// for all surrounding 9 cells
		for (int gy = -1; gy <= 1; ++gy) {
			for (int gx = -1; gx <= 1; ++gx) {
				double weight = weights[gx + 1].x * weights[gy + 1].y;

				//uint2 cell_x = math.uint2(cell_idx.x + gx - 1, cell_idx.y + gy - 1);
				Vec2<size_t> neighbCellCoords{ cellCoords.x + gx, cellCoords.y + gy };
				//float2 cell_dist = (cell_x - p.x) + 0.5f;
				Vec2<double> cellDist{ neighbCellCoords.x - p.pos.x + 0.5, neighbCellCoords.y - p.pos.y + 0.5 };
				//float2 Q = math.mul(p.C, cell_dist);
				Vec2<double> Q{ p.C.m11 * cellDist.x + p.C.m12 * cellDist.y, p.C.m21 * cellDist.x + p.C.m22 * cellDist.y };

				// scatter mass and momentum to the grid
				//int cell_index = (int)cell_x.x * grid_res + (int)cell_x.y;
				size_t neighbIdx = neighbCellCoords.x + neighbCellCoords.y * gridRes;
				//Cell cell = grid[cell_index];
				// no OOB because positions are clamped to leave space on the edges
				//Cell& cell = cellGrid[neighbIdx];
				auto& writeTup = P2G1CellGridWrites[9 * particleIdx + neighborCount];
				Cell& writeCell = std::get<2>(writeTup);
				std::get<0>(writeTup) = true;
				std::get<1>(writeTup) = neighbIdx;

				// MPM course, equation 172
				double massContrib = weight * p.mass;
				//cell.mass += massContrib;
				writeCell.mass = massContrib;

				// APIC P2G momentum contribution
				// dpX: lowercase p = MOMENTUM
				const double dpX = massContrib * (p.vel.x + Q.x);
				const double dpY = massContrib * (p.vel.y + Q.y);
				const Vec2<double> dp{ dpX, dpY };
				//cell.vel.x += dp.x;
				//cell.vel.y += dp.y;
				writeCell.vel.x = dp.x;
				writeCell.vel.y = dp.y;

				++neighborCount;
				// note: currently "cell.vel" refers to MOMENTUM, not velocity!
			}
		}
	//}
	};

	std::for_each(std::execution::par_unseq, particleArray.begin(), particleArray.end(), jobP2G);

	// can be parallelized further if indices are stored in ascending order & chunks chosen to not have same indices
	for (size_t i = 0; i < P2G1CellGridWrites.size(); ++i) {
		auto& write = P2G1CellGridWrites[i];
		size_t cellInd = std::get<1>(write);
		Cell& addCell = std::get<2>(write);
		Cell& cell = cellGrid[cellInd];
		cell.mass += addCell.mass;
		cell.vel = cell.vel + addCell.vel;
	}
}

/// <summary>
/// implements constitutive model
/// </summary>
void MLS_MPM_Fluid::JobP2G2()
{
	////std::vector<Cell> cellGridCpy = cellGrid;
	static auto jobP2G = [this](Particle& p) -> void
	{
	//for (auto it = particleArray.begin(); it < particleArray.end(); ++it)
	//{
	//	Particle& p = *it;
		size_t particleIdx = (&p - &*particleArray.begin());

		//uint2 cell_idx = (uint2)p.x;
		Vec2<size_t> cellIdx{ (size_t)p.pos.x, (size_t)p.pos.y };
		//float2 cell_diff = (p.x - cell_idx) - 0.5f;
		Vec2<double> cellDiff{ p.pos.x - cellIdx.x - 0.5, p.pos.y - cellIdx.y - 0.5 };
		// quadratic interpolation weights
		Vec2<double> weights[3];
		//weights[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
		weights[0] = { 0.5 * (0.5 - cellDiff.x) * (0.5 - cellDiff.x), 0.5 * (0.5 - cellDiff.y) * (0.5 - cellDiff.y) };
		//weights[1] = 0.75f - math.pow(cell_diff, 2);
		weights[1] = { 0.75 - cellDiff.x * cellDiff.x, 0.75 - cellDiff.y * cellDiff.y };
		//weights[2] = 0.5f * math.pow(0.5f + cell_diff, 2);
		weights[2] = { 0.5 * (0.5 + cellDiff.x) * (0.5 + cellDiff.x), 0.5 * (0.5 + cellDiff.y) * (0.5 + cellDiff.y) };

		double density = 0.0;		// for all surrounding 9 cells
		for (int gy = -1; gy <= 1; ++gy) {
			for (int gx = -1; gx <= 1; ++gx) {
				Vec2<size_t> cellX{ cellIdx.x + gx, cellIdx.y + gy };
				size_t cellIdx = cellX.x + cellX.y * gridRes;
				double weight = weights[gx + 1].x * weights[gy + 1].y;
				density += cellGrid[cellIdx].mass * weight;
			}
		}

		double volume = p.mass / density;

		// end goal, constitutive equation for isotropic fluid: 
		// stress = -pressure * I + viscosity * (velocity_gradient + velocity_gradient_transposed)

		// Tait equation of state. i clamped it as a bit of a hack.
		// clamping helps prevent particles absorbing into each other with negative pressures

		double pressure = 0.0;
		if (eosPower == 4) { // buys about 1 FPS
			double eosTerm = density * density / (restDensity * restDensity);
			eosTerm *= eosTerm;
			pressure = std::max(-0.1, eosStiffness * (eosTerm - 1));
		}
		else {
			pressure = std::max(-0.1, eosStiffness * (std::pow(density / restDensity, eosPower) - 1));
		}

		Mat2<double> stress{
			-pressure, 0.0,
			0.0, -pressure
		};

		// velocity gradient - CPIC eq. 17, where deriv of quadratic polynomial is linear
		//float2x2 dudv = p.C;
		Mat2<double> dudv = p.C;
		//float2x2 strain = dudv;
		Mat2<double> strain = dudv;

		//float trace = strain.c1.x + strain.c0.y;
		double trace = strain.m12 + strain.m21;
		//strain.c0.y = strain.c1.x = trace;
		strain.m21 = strain.m12 = trace;

		//float2x2 viscosity_term = dynamic_viscosity * strain;
		Mat2<double> viscosityTerm = dynamicViscosity * strain;
		stress = stress + viscosityTerm;

		//var eq_16_term_0 = -volume * 4 * stress * dt;
		Mat2<double> eq16Term0 = -volume * 4.0 * dt * stress;

		// for all surrounding 9 cells
		size_t neighborCount = 0;
		for (int gy = -1; gy <= 1; ++gy) {
			for (int gx = -1; gx <= 1; ++gx) {
				double weight = weights[gx + 1].x * weights[gy + 1].y;

				//uint2 cell_x = math.uint2(cell_idx.x + gx - 1, cell_idx.y + gy - 1);
				Vec2<size_t> neighbCellCoords{ cellIdx.x + gx, cellIdx.y + gy };
				//float2 cell_dist = (cell_x - p.x) + 0.5f;
				Vec2<double> cellDist{ neighbCellCoords.x - p.pos.x + 0.5, neighbCellCoords.y - p.pos.y + 0.5 };

				//int cell_index = (int)cell_x.x * grid_res + (int)cell_x.y;
				size_t neighbIdx = neighbCellCoords.x + neighbCellCoords.y * gridRes;
				//Cell cell = grid[cell_index];
				// no OOB because positions are clamped to leave space on the edges
				//Cell& cell = cellGrid[neighbIdx];
				auto& writeTup = P2G1CellGridWrites[9 * particleIdx + neighborCount];
				Cell& writeCell = std::get<2>(writeTup);
				std::get<0>(writeTup) = true;
				std::get<1>(writeTup) = neighbIdx;

				// fused force/momentum update from MLS-MPM
				// see MLS-MPM paper, equation listed after eqn. 28
				//float2 momentum = math.mul(eq_16_term_0 * weight, cell_dist);
				Vec2<double> momentum = weight * eq16Term0 * cellDist;
				//cell.vel.x += momentum.x;
				//cell.vel.y += momentum.y;
				writeCell.vel.x = momentum.x;
				writeCell.vel.y = momentum.y;

				++neighborCount;
			}
		}
	//}
	};

	std::for_each(std::execution::par_unseq, particleArray.begin(), particleArray.end(), jobP2G);

	for (size_t i = 0; i < P2G1CellGridWrites.size(); ++i) {
		auto& write = P2G1CellGridWrites[i];
		size_t cellInd = std::get<1>(write);
		Cell& addCell = std::get<2>(write);
		Cell& cell = cellGrid[cellInd];
		cell.vel = cell.vel + 0.96*addCell.vel;
	}
}


void MLS_MPM_Fluid::JobClearGrid()
{
	for (size_t i = 0; i < numCells; ++i)
	{
		Cell& cell = cellGrid[i];

		// reset grid scratch-pad entirely
		cell.mass = 0;
		cell.vel = { 0.0, 0.0 };
	}
}

void MLS_MPM_Fluid::JobUpdateGrid()
{
	for (size_t i = 0; i < numCells; ++i)
	{
		Cell& cell = cellGrid[i];

		if (cell.mass > 0)
		{
			// convert momentum to velocity, apply gravity
			cell.vel.x /= cell.mass;
			cell.vel.y /= cell.mass;

			const double dvY = dt * gravity;
			cell.vel.y += dvY;

			// 'slip' boundary conditions
			size_t x = i / gridRes;
			size_t y = i % gridRes;
			if (x < 2 || x > gridRes - 3)
				cell.vel.x = 0.0;
			if (y < 2 || y > gridRes - 3)
				cell.vel.y = 0.0;
		}
	}
}

/// <summary>
///  seems independent of constitutive model
/// </summary>
void MLS_MPM_Fluid::JobG2P()
{
	static auto jobG2P = [this](Particle& p) -> void {
		// reset particle velocity. we calculate it from scratch each step using the grid
		p.vel = { 0.0, 0.0 };

		Vec2<size_t> cellCoords{ (size_t)p.pos.x, (size_t)p.pos.y };
		Vec2<double> cellDiff{ p.pos.x - cellCoords.x - 0.5, p.pos.y - cellCoords.y - 0.5 };

		// quadratic interpolation weights
		Vec2<double> weights[3];
		weights[0] = { 0.5 * (0.5 - cellDiff.x) * (0.5 - cellDiff.x), 0.5 * (0.5 - cellDiff.y) * (0.5 - cellDiff.y) };
		weights[1] = { 0.75 - cellDiff.x * cellDiff.x, 0.75 - cellDiff.y * cellDiff.y };
		weights[2] = { 0.5 * (0.5 + cellDiff.x) * (0.5 + cellDiff.x), 0.5 * (0.5 + cellDiff.y) * (0.5 + cellDiff.y) };

		// constructing affine per-particle momentum matrix from APIC / MLS-MPM.
		// see APIC paper (https://web.archive.org/web/20190427165435/https://www.math.ucla.edu/~jteran/papers/JSSTS15.pdf), page 6
		// below equation 11 for clarification. this is calculating C = B * (D^-1) for APIC equation 8,
		// where B is calculated in the inner loop at (D^-1) = 4 is a constant when using quadratic interpolation functions
		Mat2<double> B{ 0.0, 0.0, 0.0, 0.0 };
		for (size_t gx = 0; gx < 3; ++gx)
		{
			for (size_t gy = 0; gy < 3; ++gy)
			{
				double weight = weights[gx].x * weights[gy].y;

				Vec2<size_t>cellX{ cellCoords.x + gx - 1, cellCoords.y + gy - 1 };
				size_t cellIndex = cellX.x + cellX.y * gridRes;

				Vec2<double> dist{
					cellX.x - p.pos.x + 0.5,
					cellX.y - p.pos.y + 0.5
				};
				Vec2<double> weightedVelocity{
					cellGrid[cellIndex].vel.x * weight,
					cellGrid[cellIndex].vel.y * weight
				};

				// APIC paper equation 10, constructing inner term for B
				//var term = math.float2x2(weighted_velocity * dist.x, weighted_velocity * dist.y);
				Mat2<double> term{
					weightedVelocity.x * dist.x, weightedVelocity.x * dist.y,
					weightedVelocity.y * dist.x, weightedVelocity.y * dist.y
				};

				B = B + term;

				p.vel.x += weightedVelocity.x;
				p.vel.y += weightedVelocity.y;
			}
		}
		p.C = 4.0 * B;

		// advect particles
		p.pos.x += p.vel.x * dt;
		p.pos.y += p.vel.y * dt;

		// safety clamp to ensure particles don't exit simulation domain
		p.pos.x = std::max(1.0, std::min((double)gridRes - 2, p.pos.x));
		p.pos.y = std::max(1.0, std::min((double)gridRes - 2, p.pos.y));

		//// mouse interaction
		//if (mouse_down) {
		//	var dist = p.x - mouse_pos;
		//	if (math.dot(dist, dist) < mouse_radius * mouse_radius) {
		//		float norm_factor = (math.length(dist) / mouse_radius);
		//		norm_factor = math.pow(math.sqrt(norm_factor), 8);
		//		var force = math.normalize(dist) * norm_factor * 0.5f;
		//		p.v += force;
		//	}
		//}

		// boundaries
		//float2 x_n = p.x + p.v;
		Vec2<double> xn{ p.pos.x + p.vel.x, p.pos.y + p.vel.y };
		const double wallMin = 3.0;
		double wallMax = (double)gridRes - 3.0;
		if (xn.x < wallMin) p.vel.x += wallMin - xn.x;
		if (xn.x > wallMax) p.vel.x += wallMax - xn.x;
		if (xn.y < wallMin) p.vel.y += wallMin - xn.y;
		if (xn.y > wallMax) p.vel.y += wallMax - xn.y;
	};

	std::for_each(std::execution::par_unseq, particleArray.begin(), particleArray.end(), jobG2P);
}




// main
int main()
{
	constexpr double msPerFrame = 1000.f / FPS;

	sf::RenderWindow window(sf::VideoMode(SIM_RES_PX, SIM_RES_PX), "SFML works!");
	MLS_MPM_Fluid sim(window); 

	sim.Start();

	sf::Clock clk;
	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
			{
				window.close();
				break;
			}
		}
		sim.Update();

		sf::Time elapsed = clk.getElapsedTime();
		sf::Int32 waitMs = (sf::Int32)msPerFrame - elapsed.asMilliseconds();
		if (waitMs > 0)
			sf::sleep(sf::milliseconds(waitMs));
		clk.restart();

		actualFPS = 1000 / elapsed.asMilliseconds();
	}

	std::cout << "Simulation ended" << std::endl;
	std::cin.get();
}


/*
int main()
{
	sf::RenderWindow window(sf::VideoMode(400, 400), "SFML works!");
	sf::CircleShape shape(2.f, 7);
	shape.setFillColor(sf::Color::Green);
	shape.setPosition(100.f, 50.f);

	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}

		window.clear();
		window.draw(shape);
		window.display();
	}

	return 0;
}
*/

