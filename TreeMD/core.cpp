#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

namespace py = pybind11;
using namespace pybind11::literals;

struct Particle {
    float x, y;
    float vx, vy;
};

struct Simulation {
    int N;
    float box_size, vel_init;
    float dt;
    float eps, sigma, k, G;
    bool use_gravity;
    std::vector<Particle> p;

    Simulation(int N_, float box_size_=1.0f, float vel_init_=1.0f, float dt_=1e-3f,
               float eps_=0.1f, float sigma_=0.01f, float k_=5.0f,
               float G_=0.01f, bool use_gravity_=false)
        : N(N_), box_size(box_size_), vel_init(vel_init_), dt(dt_),
          eps(eps_), sigma(sigma_), k(k_), G(G_),
          use_gravity(use_gravity_)
    {
        p.resize(N);
        for (auto &pi : p) {
            pi.x  = box_size * (rand() / (float)RAND_MAX);
            pi.y  = box_size * (rand() / (float)RAND_MAX);
            pi.vx = vel_init * (rand() / (float)RAND_MAX);
            pi.vy = vel_init * (rand() / (float)RAND_MAX);
        }
    }

    inline void lj_force(float dx, float dy, float &fx, float &fy, float &pot) const {
        if (dx > k * sigma || dy > k * sigma || dx < - k * sigma || dy < - k * sigma) {
            fx = fy = pot = 0.0f;
            return;
        }

        float r2 = dx*dx + dy*dy + 1e-10f;
        float r = std::sqrt(r2);
        float s2 = sigma * sigma / r2;
        float s6 = s2 * s2 * s2;
        float s12 = s6 * s6;

        pot = 4 * eps * (s12 - s6);

        float force = 24 * eps * (s6 - 2 * s12) / r;

        fx = force * dx / r;
        fy = force * dy / r;


    }
    /*
    inline void grav_force(float dx, float dy, float &fx, float &fy, float &pot) const {
        float r2 = dx*dx + dy*dy + 1e-6f;
        float invr = 1.0f / std::sqrt(r2);
        fx = -G * dx * invr * invr * invr;
        fy = -G * dy * invr * invr * invr;
        pot = -G * invr;
    }
    */
    py::dict step(int n_steps=1) {
        float total_E = 0.0f;

        for (int step=0; step<n_steps; ++step) {

            std::vector<float> fx(N, 0.0f), fy(N, 0.0f);

            // --- первый шаг: силы и потенциальная энергия ---
            float potential_E = 0.0f;
            for (int i=0; i<N; ++i) {
                for (int j=i+1; j<N; ++j) {
                    float dx = p[j].x - p[i].x;
                    float dy = p[j].y - p[i].y;
                    float fxi, fyi, pot_ij;

                    if (use_gravity)
                        lj_force(dx, dy, fxi, fyi, pot_ij); // + grav_force(dx, dy, fxi, fyi, pot_ij);
                    else
                        lj_force(dx, dy, fxi, fyi, pot_ij);

                    fx[i] += fxi;
                    fy[i] += fyi;
                    fx[j] -= fxi;
                    fy[j] -= fyi;

                    potential_E += pot_ij;
                }
            }

            // --- Velocity Verlet ---
            for (int i=0; i<N; ++i) {
                p[i].vx += 0.5f * fx[i] * dt;
                p[i].vy += 0.5f * fy[i] * dt;

                p[i].x += p[i].vx * dt;
                p[i].y += p[i].vy * dt;

                // отражение от стен
                if (p[i].x < 0.0f)  { p[i].x = -p[i].x;  p[i].vx *= -1.0f; }
                if (p[i].x > box_size) { p[i].x = 2*box_size - p[i].x; p[i].vx *= -1.0f; }
                if (p[i].y < 0.0f)  { p[i].y = -p[i].y;  p[i].vy *= -1.0f; }
                if (p[i].y > box_size) { p[i].y = 2*box_size - p[i].y; p[i].vy *= -1.0f; }
            }

            // --- вторая половина скорости ---
            std::vector<float> fx2(N, 0.0f), fy2(N, 0.0f);
            for (int i=0; i<N; ++i) {
                for (int j=i+1; j<N; ++j) {
                    float dx = p[j].x - p[i].x;
                    float dy = p[j].y - p[i].y;
                    float fxi, fyi, pot_ij;

                    if (use_gravity)
                        lj_force(dx, dy, fxi, fyi, pot_ij); // + grav_force(dx, dy, fxi, fyi, pot_ij);
                    else
                        lj_force(dx, dy, fxi, fyi, pot_ij);

                    fx2[i] += fxi;
                    fy2[i] += fyi;
                    fx2[j] -= fxi;
                    fy2[j] -= fyi;
                }
            }

            for (int i=0; i<N; ++i) {
                p[i].vx += 0.5f * fx2[i] * dt;
                p[i].vy += 0.5f * fy2[i] * dt;
            }

            // --- энергия ---
            float kinetic_E = 0.0f;
            for (int i=0; i<N; ++i) {
                kinetic_E += 0.5f * (p[i].vx * p[i].vx + p[i].vy * p[i].vy);
            }

            total_E = kinetic_E + potential_E;
        }

        return py::dict("E_total"_a=total_E);
    }

    py::array_t<float> get_positions() const {
        auto arr = py::array_t<float>({N, 2});
        auto buf = arr.mutable_unchecked<2>();
        for (int i=0; i<N; ++i) {
            buf(i,0) = p[i].x;
            buf(i,1) = p[i].y;
        }
        return arr;
    }
};

PYBIND11_MODULE(core, m) {
    py::class_<Simulation>(m, "Simulation")
        .def(py::init<int,float,float,float,float,float,float,float,bool>(),
             py::arg("N"), py::arg("box_size")=1.0f, py::arg("vel_init")=1.0f, py::arg("dt")=1e-3f,
             py::arg("eps")=0.1f, py::arg("sigma")=0.01f, py::arg("k")=5.0f,
             py::arg("G")=0.01f, py::arg("use_gravity")=false)
        .def("step", &Simulation::step)
        .def("get_positions", &Simulation::get_positions);
}
