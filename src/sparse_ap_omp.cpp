#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct Dataset {
    int n = 0;
    int d = 0;
    std::vector<double> x;
};

struct Options {
    std::string mode = "sparse_ap";
    std::string input_path;
    std::string output_path = "labels.csv";
    bool generate = false;
    int generate_n = 0;
    int generate_d = 2;
    int generate_k = 3;
    int neighbors = 20;
    int max_iter = 200;
    int threads = 0;
    int seed = 42;
    int verbose_every = 10;
    double damping = 0.70;
    double tol = 1e-4;
    double preference = std::numeric_limits<double>::quiet_NaN();
    double preference_quantile = 0.50;
    double spread = 8.0;
    double noise = 0.60;
};

struct SparseGraph {
    int n = 0;
    std::vector<int> row_ptr;
    std::vector<int> row_idx;
    std::vector<int> col_idx;
    std::vector<int> col_ptr;
    std::vector<int> col_edges;
    std::vector<int> diag_edge;
    std::vector<double> sim;
    double preference = 0.0;
};

struct APResult {
    int iterations = 0;
    double max_delta = 0.0;
    double objective = 0.0;
    std::vector<int> exemplars;
    std::vector<int> labels;
};

static void usage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " --input data.csv [options]\n"
        << "  " << prog << " --generate N D TRUE_K [options]\n\n"
        << "Options:\n"
        << "  --mode NAME                label used in CSV metrics output (default sparse_ap)\n"
        << "  --neighbors L              top-L neighbors per sample, excluding self (default 20)\n"
        << "  --max-iter T               max AP iterations (default 200)\n"
        << "  --damping X                damping in [0.5, 0.95], larger is stabler (default 0.70)\n"
        << "  --tol EPS                  stop when max message change < EPS (default 1e-4)\n"
        << "  --preference P             self similarity; lower P gives fewer clusters\n"
        << "  --preference-quantile Q    auto preference quantile from sparse similarities (default 0.50)\n"
        << "  --threads T                OpenMP thread count when compiled with -fopenmp\n"
        << "  --output labels.csv        output path (default labels.csv)\n"
        << "  --verbose-every T          print progress every T iterations; 0 disables (default 10)\n"
        << "  --seed S                   random seed for generated data (default 42)\n"
        << "  --spread X                 generated cluster center range (default 8.0)\n"
        << "  --noise X                  generated cluster noise stddev (default 0.60)\n";
}

static std::string trim(const std::string& s) {
    const char* ws = " \t\r\n";
    const size_t first = s.find_first_not_of(ws);
    if (first == std::string::npos) return "";
    const size_t last = s.find_last_not_of(ws);
    return s.substr(first, last - first + 1);
}

static double parse_double(const std::string& s) {
    size_t pos = 0;
    const double v = std::stod(trim(s), &pos);
    if (trim(s).substr(pos).empty()) return v;
    throw std::invalid_argument("bad number: " + s);
}

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const std::string& flag) -> char* {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value after " + flag);
            }
            return argv[++i];
        };

        if (a == "--help" || a == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (a == "--mode") {
            opt.mode = need(a);
        } else if (a == "--input") {
            opt.input_path = need(a);
        } else if (a == "--generate") {
            if (i + 3 >= argc) throw std::runtime_error("--generate needs N D TRUE_K");
            opt.generate = true;
            opt.generate_n = std::stoi(argv[++i]);
            opt.generate_d = std::stoi(argv[++i]);
            opt.generate_k = std::stoi(argv[++i]);
        } else if (a == "--neighbors") {
            opt.neighbors = std::stoi(need(a));
        } else if (a == "--max-iter") {
            opt.max_iter = std::stoi(need(a));
        } else if (a == "--damping") {
            opt.damping = parse_double(need(a));
        } else if (a == "--tol") {
            opt.tol = parse_double(need(a));
        } else if (a == "--preference") {
            opt.preference = parse_double(need(a));
        } else if (a == "--preference-quantile") {
            opt.preference_quantile = parse_double(need(a));
        } else if (a == "--threads") {
            opt.threads = std::stoi(need(a));
        } else if (a == "--output") {
            opt.output_path = need(a);
        } else if (a == "--verbose-every") {
            opt.verbose_every = std::stoi(need(a));
        } else if (a == "--seed") {
            opt.seed = std::stoi(need(a));
        } else if (a == "--spread") {
            opt.spread = parse_double(need(a));
        } else if (a == "--noise") {
            opt.noise = parse_double(need(a));
        } else {
            throw std::runtime_error("unknown option: " + a);
        }
    }

    if (!opt.generate && opt.input_path.empty()) {
        throw std::runtime_error("provide --input data.csv or --generate N D TRUE_K");
    }
    if (opt.neighbors < 0) throw std::runtime_error("--neighbors must be non-negative");
    if (opt.max_iter <= 0) throw std::runtime_error("--max-iter must be positive");
    if (opt.damping < 0.0 || opt.damping >= 1.0) {
        throw std::runtime_error("--damping must be in [0, 1)");
    }
    if (opt.preference_quantile < 0.0 || opt.preference_quantile > 1.0) {
        throw std::runtime_error("--preference-quantile must be in [0, 1]");
    }
    return opt;
}

static Dataset load_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open input file: " + path);

    Dataset data;
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        std::vector<double> row;
        try {
            if (line.find(',') != std::string::npos) {
                std::stringstream ss(line);
                std::string cell;
                while (std::getline(ss, cell, ',')) {
                    if (!trim(cell).empty()) row.push_back(parse_double(cell));
                }
            } else {
                std::stringstream ss(line);
                std::string cell;
                while (ss >> cell) row.push_back(parse_double(cell));
            }
        } catch (const std::exception&) {
            throw std::runtime_error("non-numeric value at line " + std::to_string(line_no));
        }

        if (row.empty()) continue;
        if (data.d == 0) {
            data.d = static_cast<int>(row.size());
        } else if (static_cast<int>(row.size()) != data.d) {
            throw std::runtime_error("inconsistent dimension at line " + std::to_string(line_no));
        }
        data.x.insert(data.x.end(), row.begin(), row.end());
        ++data.n;
    }

    if (data.n == 0) throw std::runtime_error("input file has no numeric rows");
    return data;
}

static Dataset generate_blobs(const Options& opt) {
    if (opt.generate_n <= 0 || opt.generate_d <= 0 || opt.generate_k <= 0) {
        throw std::runtime_error("--generate values must be positive");
    }

    std::mt19937 rng(opt.seed);
    std::uniform_real_distribution<double> center_dist(-opt.spread, opt.spread);
    std::normal_distribution<double> noise_dist(0.0, opt.noise);

    std::vector<double> centers(opt.generate_k * opt.generate_d);
    for (double& v : centers) v = center_dist(rng);

    Dataset data;
    data.n = opt.generate_n;
    data.d = opt.generate_d;
    data.x.resize(static_cast<size_t>(data.n) * data.d);

    for (int i = 0; i < data.n; ++i) {
        const int c = i % opt.generate_k;
        for (int j = 0; j < data.d; ++j) {
            data.x[static_cast<size_t>(i) * data.d + j] =
                centers[static_cast<size_t>(c) * data.d + j] + noise_dist(rng);
        }
    }
    return data;
}

static inline double dist2(const Dataset& data, int i, int j) {
    const double* xi = &data.x[static_cast<size_t>(i) * data.d];
    const double* xj = &data.x[static_cast<size_t>(j) * data.d];
    double s = 0.0;
    for (int t = 0; t < data.d; ++t) {
        const double diff = xi[t] - xj[t];
        s += diff * diff;
    }
    return s;
}

static double quantile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    const size_t idx = static_cast<size_t>(std::floor(q * (values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + static_cast<long>(idx), values.end());
    return values[idx];
}

static SparseGraph build_sparse_knn_graph(const Dataset& data, const Options& opt) {
    const int n = data.n;
    const int L = std::min(opt.neighbors, std::max(0, n - 1));
    std::vector<std::vector<std::pair<int, double>>> rows(n);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        std::vector<std::pair<double, int>> dists;
        dists.reserve(std::max(0, n - 1));
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            dists.emplace_back(dist2(data, i, j), j);
        }

        if (L > 0) {
            if (L < static_cast<int>(dists.size())) {
                std::nth_element(dists.begin(), dists.begin() + L, dists.end());
            }
            std::sort(dists.begin(), dists.begin() + L);
        }

        rows[i].reserve(static_cast<size_t>(L) + 1);
        rows[i].push_back({i, 0.0});
        for (int t = 0; t < L; ++t) {
            rows[i].push_back({dists[t].second, -dists[t].first});
        }
    }

    std::vector<double> candidate_similarities;
    candidate_similarities.reserve(static_cast<size_t>(n) * std::max(1, L));
    for (int i = 0; i < n; ++i) {
        for (size_t p = 1; p < rows[i].size(); ++p) {
            candidate_similarities.push_back(rows[i][p].second);
        }
    }

    SparseGraph g;
    g.n = n;
    g.preference = std::isnan(opt.preference)
                       ? quantile(candidate_similarities, opt.preference_quantile)
                       : opt.preference;

    for (int i = 0; i < n; ++i) {
        rows[i][0].second = g.preference;
    }

    g.row_ptr.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        g.row_ptr[i + 1] = g.row_ptr[i] + static_cast<int>(rows[i].size());
    }
    const int edges = g.row_ptr[n];
    g.row_idx.resize(edges);
    g.col_idx.resize(edges);
    g.sim.resize(edges);
    g.diag_edge.assign(n, -1);

    for (int i = 0; i < n; ++i) {
        for (int p = g.row_ptr[i]; p < g.row_ptr[i + 1]; ++p) {
            const auto& e = rows[i][static_cast<size_t>(p - g.row_ptr[i])];
            g.row_idx[p] = i;
            g.col_idx[p] = e.first;
            g.sim[p] = e.second;
            if (e.first == i) g.diag_edge[i] = p;
        }
    }

    std::vector<int> col_count(n, 0);
    for (int k : g.col_idx) ++col_count[k];
    g.col_ptr.assign(n + 1, 0);
    for (int k = 0; k < n; ++k) {
        g.col_ptr[k + 1] = g.col_ptr[k] + col_count[k];
    }
    g.col_edges.resize(edges);
    std::vector<int> next = g.col_ptr;
    for (int e = 0; e < edges; ++e) {
        const int k = g.col_idx[e];
        g.col_edges[next[k]++] = e;
    }

    return g;
}

static int count_exemplars(const SparseGraph& g,
                           const std::vector<double>& r,
                           const std::vector<double>& a) {
    int count = 0;
    for (int k = 0; k < g.n; ++k) {
        const int e = g.diag_edge[k];
        if (a[e] + r[e] > 0.0) ++count;
    }
    return count;
}

static APResult run_sparse_ap(const Dataset& data, const SparseGraph& g, const Options& opt) {
    const int n = g.n;
    const int edges = static_cast<int>(g.sim.size());
    const double keep = opt.damping;
    const double take = 1.0 - opt.damping;
    std::vector<double> r(edges, 0.0);
    std::vector<double> a(edges, 0.0);

    APResult result;
    for (int iter = 1; iter <= opt.max_iter; ++iter) {
        double max_delta = 0.0;

#pragma omp parallel for schedule(static) reduction(max:max_delta)
        for (int i = 0; i < n; ++i) {
            double best1 = -std::numeric_limits<double>::infinity();
            double best2 = -std::numeric_limits<double>::infinity();
            int best_edge = -1;

            for (int p = g.row_ptr[i]; p < g.row_ptr[i + 1]; ++p) {
                const double v = a[p] + g.sim[p];
                if (v > best1) {
                    best2 = best1;
                    best1 = v;
                    best_edge = p;
                } else if (v > best2) {
                    best2 = v;
                }
            }

            for (int p = g.row_ptr[i]; p < g.row_ptr[i + 1]; ++p) {
                double max_except = (p == best_edge) ? best2 : best1;
                if (!std::isfinite(max_except)) max_except = 0.0;
                const double new_r = g.sim[p] - max_except;
                const double damped = keep * r[p] + take * new_r;
                max_delta = std::max(max_delta, std::abs(damped - r[p]));
                r[p] = damped;
            }
        }

#pragma omp parallel for schedule(static) reduction(max:max_delta)
        for (int k = 0; k < n; ++k) {
            const int diag = g.diag_edge[k];
            const double rkk = r[diag];
            double sum_pos = 0.0;

            for (int q = g.col_ptr[k]; q < g.col_ptr[k + 1]; ++q) {
                const int e = g.col_edges[q];
                if (g.row_idx[e] != k && r[e] > 0.0) {
                    sum_pos += r[e];
                }
            }

            for (int q = g.col_ptr[k]; q < g.col_ptr[k + 1]; ++q) {
                const int e = g.col_edges[q];
                double new_a = 0.0;
                if (g.row_idx[e] == k) {
                    new_a = sum_pos;
                } else {
                    new_a = std::min(0.0, rkk + sum_pos - std::max(0.0, r[e]));
                }

                const double damped = keep * a[e] + take * new_a;
                max_delta = std::max(max_delta, std::abs(damped - a[e]));
                a[e] = damped;
            }
        }

        result.iterations = iter;
        result.max_delta = max_delta;

        if (opt.verbose_every > 0 &&
            (iter == 1 || iter % opt.verbose_every == 0 || max_delta < opt.tol)) {
            std::cerr << "iter=" << std::setw(4) << iter
                      << " max_delta=" << std::scientific << std::setprecision(3) << max_delta
                      << " exemplars=" << count_exemplars(g, r, a) << "\n";
        }

        if (iter >= 5 && max_delta < opt.tol) break;
    }

    double best_diag = -std::numeric_limits<double>::infinity();
    int fallback_exemplar = 0;
    for (int k = 0; k < n; ++k) {
        const int e = g.diag_edge[k];
        const double score = a[e] + r[e];
        if (score > 0.0) result.exemplars.push_back(k);
        if (score > best_diag) {
            best_diag = score;
            fallback_exemplar = k;
        }
    }
    if (result.exemplars.empty()) result.exemplars.push_back(fallback_exemplar);

    result.labels.assign(n, 0);
    double objective = 0.0;
#pragma omp parallel for schedule(static) reduction(+:objective)
    for (int i = 0; i < n; ++i) {
        double best = std::numeric_limits<double>::infinity();
        int label = 0;
        for (int c = 0; c < static_cast<int>(result.exemplars.size()); ++c) {
            const double d = dist2(data, i, result.exemplars[c]);
            if (d < best) {
                best = d;
                label = c;
            }
        }
        result.labels[i] = label;
        objective += best;
    }
    result.objective = objective;
    return result;
}

static void write_labels(const std::string& path, const APResult& result) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write output file: " + path);
    out << "sample,label,exemplar\n";
    for (int i = 0; i < static_cast<int>(result.labels.size()); ++i) {
        const int label = result.labels[i];
        out << i << ',' << label << ',' << result.exemplars[label] << '\n';
    }
}

static int active_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

#ifdef _OPENMP
        if (opt.threads > 0) omp_set_num_threads(opt.threads);
#else
        if (opt.threads > 1) {
            std::cerr << "warning: binary was compiled without OpenMP; running serially\n";
        }
#endif

        const Dataset data = opt.generate ? generate_blobs(opt) : load_csv(opt.input_path);
        std::cerr << "samples=" << data.n << " dim=" << data.d << "\n";

        auto t0 = std::chrono::steady_clock::now();
        const SparseGraph graph = build_sparse_knn_graph(data, opt);
        auto t1 = std::chrono::steady_clock::now();
        const APResult result = run_sparse_ap(data, graph, opt);
        auto t2 = std::chrono::steady_clock::now();

        const double build_s =
            std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        const double ap_s =
            std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        const double total_s =
            std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t0).count();

        write_labels(opt.output_path, result);

        std::cout << opt.mode << ','
                  << data.n << ','
                  << data.d << ','
                  << opt.neighbors << ','
                  << ictive_threads() << ','
                  << std::setprecision(10) << build_s << ','
                  << ap_s << ','
                  << total_s << ','
                  << result.iterations << ','
                  << result.exemplars.size() << ','
                  << result.objective << ','
                  << graph.sim.size() << ','
                  << graph.preference << ','
                  << opt.damping << ','
                  << result.max_delta << '\n';

        std::cerr << "edges=" << graph.sim.size()
                  << " preference=" << graph.preference
                  << " clusters=" << result.exemplars.size()
                  << " iterations=" << result.iterations
                  << " objective=" << result.objective << "\n"
                  << "build_seconds=" << build_s
                  << " ap_seconds=" << ap_s
                  << " output=" << opt.output_path << "\n";

#ifdef _OPENMP
        std::cerr << "openmp_threads=" << omp_get_max_threads() << "\n";
#else
        std::cerr << "openmp_threads=1 (OpenMP disabled at compile time)\n";
#endif
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n";
        usage(argv[0]);
        return 1;
    }
}
