// MPM_guide.cpp
//
#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <random>

#define GRID_RES 64
#define SIM_RES_PX 400
constexpr double gridToSimFactor = SIM_RES_PX / GRID_RES;

template<typename T>
struct Vec2
{
	T x;
	T y;
};

template<typename T>
struct Mat2
{
	T m11, m12, m21, m22;
};

Vec2<size_t> WorldCoordsToPxCoords(Vec2<double> wc)
{
	return Vec2<size_t>{(size_t)(gridToSimFactor * wc.x + 0.5), (SIM_RES_PX - (size_t)(gridToSimFactor * wc.y + 0.5))};
}

struct Particle
{
	Vec2<double> pos; // position
	Vec2<double> vel; // velocity
	double mass;
	double padding; // for performance
	Mat2<double> C; // affine momentum matrix
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
	void ShowNextFrame(const std::vector<Particle>& particles); // better interface
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
	auto& cs = window.getSettings();

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
	return start + (end-start)*urand(gen);
}

class MLS_MPM_Intro
{
public:
	MLS_MPM_Intro(sf::RenderWindow& window);
	virtual ~MLS_MPM_Intro();
	void Start();
	void Update();
	void HandleMouseInteraction();
	void Simulate();

	constexpr static size_t gridRes = GRID_RES;
	constexpr static size_t numCells = gridRes * gridRes;

	constexpr static double dt = 1.0;
	constexpr static int iterations = static_cast<int>(1.0 / dt);

	constexpr static double gravity = -0.05;

	size_t numParticles;
	std::vector<Particle> particleArray;
	Cell* cellGrid;

	// quadratic interpolation weights
	Vec2<double> weights[3];

	sf::RenderWindow& window;
	Renderer renderer;

	const double mouseRadius = 10.0;
	bool mouseDown = false;
	Vec2<double> mousePos{ 0.0, 0.0 };
};

MLS_MPM_Intro::MLS_MPM_Intro(sf::RenderWindow& window) :
	window(window),
	renderer(window),
	cellGrid(nullptr),
	numParticles(0),
	weights{ 0, 0 }
{
}

MLS_MPM_Intro::~MLS_MPM_Intro()
{
	if (cellGrid)
		delete[] cellGrid;
}

void MLS_MPM_Intro::Start()
{
	std::uniform_real_distribution<> rand(0.0, 1.0);
	// initialize a bunch of points in a square
	std::vector<Vec2<double>> tempPositions;
	const double spacing = 1.0f;
	const int boxX = 16;
	const int boxY = 16;
	const double sx = gridRes / 2.0;
	const double sy = gridRes / 2.0;
	for (double i = sx - boxX / 2; i < sx + boxX / 2; i += spacing)
	{
		for (double j = sy - boxY / 2; j < sy + boxY / 2; j += spacing)
		{
			Vec2<double> pos{ i, j };
			tempPositions.push_back(pos);
		}
	}

	numParticles = tempPositions.size();

	// populate our array of particles, set their initial state
	particleArray.resize(numParticles);
	for (size_t i = 0; i < numParticles; ++i)
	{
		Particle p{
			tempPositions[i],
			Vec2<double>{uRandRange(-0.25, 0.25), uRandRange(2.25, 3.25) / 2},
			1.0,
			0.0
		};
		particleArray[i] = p;
	}

	cellGrid = new Cell[numCells];
}

void MLS_MPM_Intro::Update()
{
	HandleMouseInteraction();

	for (int i = 0; i < iterations; ++i) 
	{
		Simulate();
	}

	renderer.ShowNextFrame(particleArray);
}

void MLS_MPM_Intro::HandleMouseInteraction()
{
	return;
}

void MLS_MPM_Intro::Simulate()
{
	// reset grid scratchpad
	for (size_t i = 0; i < numCells; ++i) 
	{
		Cell& cell = cellGrid[i];
		cell.mass = 0;
		cell.vel = { 0, 0 };
	}

	// P2G
	for (size_t i = 0; i < numParticles; ++i) 
	{
		Particle& p = particleArray[i];

		// quadratic interpolation weights
		Vec2<size_t> cellIdx{ (size_t)p.pos.x, (size_t)p.pos.y };
		Vec2<double> cellDiff{ p.pos.x - cellIdx.x - 0.5, p.pos.y - cellIdx.y - 0.5 };
		weights[0] = { 0.5 * std::pow(0.5 - cellDiff.x, 2.0), 0.5 * std::pow(0.5 - cellDiff.y, 2.0) };
		weights[1] = { 0.75 - std::pow(cellDiff.x, 2.0), 0.75 - std::pow(cellDiff.y, 2.0) };
		weights[2] = { 0.5 * std::pow(0.5 + cellDiff.x, 2.0), 0.5 * std::pow(0.5 + cellDiff.y, 2.0) };

		// for all surrounding 9 cells
		for (size_t gx = 0; gx < 3; ++gx) 
		{
			for (size_t gy = 0; gy < 3; ++gy) 
			{
				double weight = weights[gx].x * weights[gy].y;

				Vec2<size_t> cellX{ cellIdx.x + gx - 1, cellIdx.y + gy - 1 };
				Vec2<double> cellDist{ cellX.x - p.pos.x + 0.5, cellX.y - p.pos.y + 0.5 };
				// 2x2 matrix * 2x1 vector
				Vec2<double> Q = { p.C.m11 * cellDist.x + p.C.m12 * cellDist.y, p.C.m21 * cellDist.x + p.C.m22 * cellDist.y };

				// MPM course, equation 172
				double massContrib = weight * p.mass;

				// converting 2D index to 1D
				size_t cellIndex = cellX.x * gridRes + cellX.y;
				Cell& cell = cellGrid[cellIndex];

				// scatter mass to the grid
				cell.mass += massContrib;

				// note: here, "cell.vel" refers to MOMENTUM, not velocity!
				// this gets corrected in the UpdateGrid step further down.
				cell.vel.x += massContrib * (p.vel.x + Q.x);
				cell.vel.y += massContrib * (p.vel.y + Q.y);
			}
		}
	}

	// grid velocity update
	for (size_t i = 0; i < numCells; ++i) 
	{
		Cell& cell = cellGrid[i];

		if (cell.mass > 0) 
		{
			// convert momentum to velocity, apply gravity
			cell.vel.x /= cell.mass;
			cell.vel.y /= cell.mass;
			cell.vel.y += dt * gravity;

			// boundary conditions
			size_t x = i / gridRes;
			size_t y = i % gridRes;
			if (x < 2 || x > gridRes - 3) 
				cell.vel.x = 0;
			if (y < 2 || y > gridRes - 3) 
				cell.vel.y = 0;
		}

		cellGrid[i] = cell;
	}

	// G2P
	for (size_t i = 0; i < numParticles; ++i) 
	{
		Particle& p = particleArray[i];

		// reset particle velocity. we calculate it from scratch each step using the grid
		p.vel = { 0.0, 0.0 };

		// quadratic interpolation weights
		Vec2<size_t> cellIdx{ (size_t)p.pos.x, (size_t)p.pos.y };
		Vec2<double> cellDiff{ p.pos.x - cellIdx.x - 0.5, p.pos.y - cellIdx.y - 0.5 };

		weights[0] = { 0.5 * std::pow(0.5 - cellDiff.x, 2.0), 0.5 * std::pow(0.5 - cellDiff.y, 2.0) };
		weights[1] = { 0.75 - std::pow(cellDiff.x, 2.0), 0.75 - std::pow(cellDiff.y, 2.0) };
		weights[2] = { 0.5 * std::pow(0.5 + cellDiff.x, 2.0), 0.5 * std::pow(0.5 + cellDiff.y, 2.0) };

		// constructing affine per-particle momentum matrix from APIC / MLS-MPM.
		// see APIC paper (https://web.archive.org/web/20190427165435/https://www.math.ucla.edu/~jteran/papers/JSSTS15.pdf), page 6
		// below equation 11 for clarification. this is calculating C = B * (D^-1) for APIC equation 8,
		// where B is calculated in the inner loop at (D^-1) = 4 is a constant when using quadratic interpolation functions
		Mat2<double> B = { 0.0, 0.0,
						0.0, 0.0 };
		for (size_t gx = 0; gx < 3; ++gx) 
		{
			for (size_t gy = 0; gy < 3; ++gy) 
			{
				double weight = weights[gx].x * weights[gy].y;

				Vec2<size_t> cellX{ cellIdx.x + gx - 1, cellIdx.y + gy - 1 };
				size_t cellIdx = cellX.x * gridRes + cellX.y;

				Vec2<double> dist{ cellX.x - p.pos.x + 0.5, cellX.y - p.pos.y + 0.5 };
				Vec2<double> weightedVelocity{ cellGrid[cellIdx].vel.x * weight, cellGrid[cellIdx].vel.y * weight };

				// APIC paper equation 10, constructing inner term for B
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
		p.C = {
			B.m11 * 4, B.m12 * 4,
			B.m21 * 4, B.m22 * 4
		};

		// advect particles
		p.pos.x += p.vel.x * dt;
		p.pos.y += p.vel.y * dt;

		// safety clamp to ensure particles don't exit simulation domain
		p.pos.x = std::min(std::max(1.0, p.pos.x), gridRes - 2.0);
		p.pos.y = std::min(std::max(1.0, p.pos.y), gridRes - 2.0);

		// mouse interaction
		//if (mouse_down) {
		//	var dist = p.x - mouse_pos;
		//	if (math.dot(dist, dist) < mouse_radius * mouse_radius) {
		//		float norm_factor = (math.length(dist) / mouse_radius);
		//		norm_factor = math.pow(math.sqrt(norm_factor), 8);
		//		var force = math.normalize(dist) * norm_factor * 0.5f;
		//		p.v += force;
		//	}
		//}
	}
}

// main
int main()
{
	constexpr size_t FPS = 20;
	constexpr double msPerFrame = 1000.f / FPS;

	sf::RenderWindow window(sf::VideoMode(400, 400), "SFML works!");
	MLS_MPM_Intro sim(window);

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
