// MPM_guide_elastic_solid.cpp
//
#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <execution>
#include <mutex>

constexpr size_t GRID_RES = 64;
constexpr size_t SIM_RES_PX = 400;
constexpr double gridToSimFactor = SIM_RES_PX / GRID_RES;
constexpr size_t division = 16; // multithreading
constexpr static double dt = 0.01;
constexpr static double c = 0.25 / dt; // "speed of light"

namespace Math
{
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
	return Vec2<size_t>{(size_t)(gridToSimFactor* wc.x + 0.5), (SIM_RES_PX - (size_t)(gridToSimFactor * wc.y + 0.5))};
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
	double RelativisticMass()
	{
		return mass / std::sqrt(1 - vel.Mag() * vel.Mag() / (c * c));
	}

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
	void ShowNextFrame(const std::vector<Particle>& particles);
	void AddParticle(const Particle& p);
private:
	std::vector<sf::CircleShape> shapes;
	sf::RenderWindow& window;
};

Renderer::Renderer(sf::RenderWindow& win) :
	window(win)
{
}

void Renderer::ShowNextFrame(const std::vector<Particle>& particles)
{
	window.clear(sf::Color(66, 189, 245));

	for (auto it = particles.begin(); it < particles.end(); ++it)
		AddParticle(*it);

	for (sf::Shape& shape : shapes)
		window.draw(shape);

	shapes.clear();

	window.display();
}

void Renderer::AddParticle(const Particle& p)
{
	sf::CircleShape circle(1.4f, 6);
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
	constexpr static double restDensity = 4.0;
	constexpr static double dynamicViscosity = 0.1;
	// equation of state
	constexpr static double eosStiffness = 20.0; // compressability
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
	SpawnBox((int)(15/32.0 * gridRes), (int)(5/8.0 * gridRes), 32, 32);
	numParticles = tempPositions.size();

	particleArray.resize(numParticles);

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

	renderer.ShowNextFrame(particleArray);
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
	const double spacing = 0.5;
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
void MLS_MPM_Fluid::JobP2G1()
{
	////std::vector<Cell> cellGridCpy = cellGrid;
	//static auto jobP2G = [this](Particle& p) -> void
	//{
	for (auto it = particleArray.begin(); it < particleArray.end(); ++it)
	{
		Particle& p = *it;

		Vec2<size_t> cellIdx{ (size_t)p.pos.x, (size_t)p.pos.y };
		Vec2<double> cellDiff{ p.pos.x - cellIdx.x - 0.5, p.pos.y - cellIdx.y - 0.5 };

		Vec2<double> weights[3];
		//weights[0] = 0.5f * math.pow(0.5f - cell_diff, 2);
		weights[0] = { 0.5 * (0.5 - cellDiff.x) * (0.5 - cellDiff.x), 0.5 * (0.5 - cellDiff.y) * (0.5 - cellDiff.y) };
		//weights[1] = 0.75f - math.pow(cell_diff, 2);
		weights[1] = { 0.75 - cellDiff.x * cellDiff.x, 0.75 - cellDiff.y * cellDiff.y };
		//weights[2] = 0.5f * math.pow(0.5f + cell_diff, 2);
		weights[2] = { 0.5 * (0.5 + cellDiff.x) * (0.5 + cellDiff.x), 0.5 * (0.5 + cellDiff.y) * (0.5 + cellDiff.y) };

		Mat2<double> C = p.C;

		// for all surrounding 9 cells
		for (size_t gx = 0; gx < 3; ++gx)
		{
			for (size_t gy = 0; gy < 3; ++gy)
			{
				double weight = weights[gx].x * weights[gy].y;

				//uint2 cell_x = math.uint2(cell_idx.x + gx - 1, cell_idx.y + gy - 1);
				Vec2<size_t> cellX{ cellIdx.x + gx - 1, cellIdx.y + gy - 1 };
				//float2 cell_dist = (cell_x - p.x) + 0.5f;
				Vec2<double> cellDist{ cellX.x - p.pos.x + 0.5, cellX.y - p.pos.y + 0.5 };
				//float2 Q = math.mul(p.C, cell_dist);
				Vec2<double> Q{ p.C.m11 * cellDist.x + p.C.m12 * cellDist.y, p.C.m21 * cellDist.x + p.C.m22 * cellDist.y };

				// scatter mass and momentum to the grid
				//int cell_index = (int)cell_x.x * grid_res + (int)cell_x.y;
				size_t cellIdx = cellX.x * gridRes + cellX.y;
				//Cell cell = grid[cell_index];
				Cell& cell = cellGrid[cellIdx];

				// MPM course, equation 172
				double massContrib = weight * p.mass;
				cell.mass += massContrib;

				// APIC P2G momentum contribution
				// dpX: lowercase p = MOMENTUM
				const double dpX = massContrib * (p.vel.x + Q.x);
				const double dpY = massContrib * (p.vel.y + Q.y);
				const Vec2<double> dp{ dpX, dpY };
				// relativistic correction
				//dp = (1.0 / (1.0 + (1.0 / c / c) * cell.vel * dp))*(cell.vel + dp) + -1.0 * cell.vel;
				cell.vel.x += dp.x;
				cell.vel.y += dp.y;

				// note: currently "cell.vel" refers to MOMENTUM, not velocity!
			}
		}
	}
	//};

	//std::for_each(std::execution::seq, particleArray.begin(), particleArray.end(), jobP2G);
}

void MLS_MPM_Fluid::JobP2G2()
{
	////std::vector<Cell> cellGridCpy = cellGrid;
	//static auto jobP2G = [this](Particle& p) -> void
	//{
	for (auto it = particleArray.begin(); it < particleArray.end(); ++it)
	{
		Particle& p = *it;

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
		for (size_t gx = 0; gx < 3; ++gx)
		{
			for (size_t gy = 0; gy < 3; ++gy)
			{
				Vec2<size_t> cellX{ cellIdx.x + gx - 1, cellIdx.y + gy - 1 };
				size_t cellIdx = cellX.x * gridRes + cellX.y;
				double weight = weights[gx].x * weights[gy].y;
				density += cellGrid[cellIdx].mass * weight;
			}
		}

		double volume = p.mass / density;
		
		// end goal, constitutive equation for isotropic fluid: 
		// stress = -pressure * I + viscosity * (velocity_gradient + velocity_gradient_transposed)

		// Tait equation of state. i clamped it as a bit of a hack.
		// clamping helps prevent particles absorbing into each other with negative pressures

		double pressure = std::max(-0.1, eosStiffness * (std::pow(density / restDensity, eosPower) - 1));

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
		for (size_t gx = 0; gx < 3; ++gx)
		{
			for (size_t gy = 0; gy < 3; ++gy)
			{
				double weight = weights[gx].x * weights[gy].y;

				//uint2 cell_x = math.uint2(cell_idx.x + gx - 1, cell_idx.y + gy - 1);
				Vec2<size_t> cellX{ cellIdx.x + gx - 1, cellIdx.y + gy - 1 };
				//float2 cell_dist = (cell_x - p.x) + 0.5f;
				Vec2<double> cellDist{ cellX.x - p.pos.x + 0.5, cellX.y - p.pos.y + 0.5 };

				//int cell_index = (int)cell_x.x * grid_res + (int)cell_x.y;
				size_t cellIdx = cellX.x * gridRes + cellX.y;
				//Cell cell = grid[cell_index];
				Cell& cell = cellGrid[cellIdx];

				// fused force/momentum update from MLS-MPM
				// see MLS-MPM paper, equation listed after eqn. 28
				//float2 momentum = math.mul(eq_16_term_0 * weight, cell_dist);
				Vec2<double> momentum = weight * eq16Term0 * cellDist;
				// relativistic correction
				//momentum = (1.0 / (1.0 + (1.0 / c / c) * cell.vel * momentum)) * (cell.vel + momentum) + -1.0 * cell.vel;
				cell.vel.x += momentum.x;
				cell.vel.y += momentum.y;
			}
		}
	}
	//};

	//std::for_each(std::execution::seq, particleArray.begin(), particleArray.end(), jobP2G);
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

void MLS_MPM_Fluid::JobG2P()
{
	static auto jobG2P = [this](Particle& p) -> void {
		// reset particle velocity. we calculate it from scratch each step using the grid
		p.vel = { 0.0, 0.0 };

		Vec2<size_t> cellIdx{ (size_t)p.pos.x, (size_t)p.pos.y };
		Vec2<double> cellDiff{ p.pos.x - cellIdx.x - 0.5, p.pos.y - cellIdx.y - 0.5 };

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

				Vec2<size_t>cellX{ cellIdx.x + gx - 1, cellIdx.y + gy - 1 };
				size_t cellIndex = cellX.x * gridRes + cellX.y;

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
		double wallMax = (double)gridRes - 4.0;
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
	constexpr size_t FPS = 40;
	constexpr double msPerFrame = 1000.f / FPS;

	sf::RenderWindow window(sf::VideoMode(400, 400), "SFML works!");
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
	}

	std::cout << "Simulation ended" << std::endl;
	std::cin.get();
}

