#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <tuple>
#include <vector>

namespace py = pybind11;

template <typename T>
struct hash_eigen {
    std::size_t operator()(T const &matrix) const {
        size_t seed = 0;
        for (int i = 0; i < (int)matrix.size(); i++) {
            auto elem = *(matrix.data() + i);
            seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 +
                    (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

double CalcMHWScore(std::vector<double> &scores) {
    size_t size = scores.size();
    if (size == 0) {
        return 0;  // Undefined, really.
    } else {
        sort(scores.begin(), scores.end());
        if (size % 2 == 0) {
            return (scores[size / 2 - 1] + scores[size / 2]) / 2;
        } else {
            return scores[size / 2];
        }
    }
}

class AccumulatedIDX {
   public:
    void AddIDX(const int idx) { idx_.push_back(idx); }

   public:
    std::vector<int> idx_;
};

class AccumulatedPoint {
   public:
    void AddPoint(const Eigen::Vector3d &point, const Eigen::Vector3d &color) {
        point_ += point;
        color_ += color;
        points_in_cell_.push_back(point);
        num_of_points_++;
    }

    void AddPoint(const Eigen::Vector3d &point,
                  const Eigen::Vector3d &color,
                  const Eigen::VectorXd &feat) {
        point_ += point;
        color_ += color;
        points_in_cell_.push_back(point);
        num_of_points_++;
    }

    void AddPoint(const Eigen::Vector3d &point) {
        point_ += point;
        points_in_cell_.push_back(point);
        num_of_points_++;
    }

    Eigen::Vector3d GetAveragePoint() const {
        return point_ / double(num_of_points_);
    }

    Eigen::Vector3d GetMedianPoint() const {
        std::vector<double> x;
        std::vector<double> y;
        std::vector<double> z;
        for (size_t i = 0; i < points_in_cell_.size(); i++) {
            x.push_back(points_in_cell_[i][0]);
            y.push_back(points_in_cell_[i][1]);
            z.push_back(points_in_cell_[i][2]);
        }
        double med_x = CalcMHWScore(x);
        double med_y = CalcMHWScore(y);
        double med_z = CalcMHWScore(z);
        Eigen::Vector3d med(med_x, med_y, med_z);
        return med;
    }

    Eigen::Vector3d GetAverageColor() const {
        if (num_of_points_ > 0) {
            return color_ / double(num_of_points_);
        } else {
            return color_;
        }
    }

   public:
    int num_of_points_ = 0;
    Eigen::Vector3d point_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d color_ = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> points_in_cell_;
};

class AccumulatedPointLabel : public AccumulatedPoint {
   public:
    void AddPoint(const Eigen::Vector3d &point, const Eigen::Vector3d &color, int prediction) {
        point_ += point;
        color_ += color;
        points_in_cell_.push_back(point);
        num_of_points_++;
        predictions_.push_back(prediction);
    }

    int VoteLabel() {
        std::unordered_map<int, int> map;
        for (size_t i = 0; i < predictions_.size(); i++) {
            map[predictions_[i]]++;
        }
        int max_app = 0;
        int new_label = 0;
        for (auto pair : map) {
            if (pair.second > max_app) {
                new_label = pair.first;
                max_app = pair.second;
            }
        }
        return new_label;
    }

   public:
    std::vector<int> predictions_;
};

Eigen::MatrixXd VoxelDownSample(Eigen::Matrix<double, Eigen::Dynamic, 3> &points,
                                double voxel_size) {
    if (voxel_size <= 0.0) {
        std::cout << "[VoxelDownSample] voxel_size <= 0." << std::endl;
    }
    Eigen::Vector3d voxel_size3 = Eigen::Vector3d(voxel_size, voxel_size, voxel_size);
    Eigen::Vector3d maxVal = points.colwise().maxCoeff();
    Eigen::Vector3d minVal = points.colwise().minCoeff();
    Eigen::Vector3d voxel_max_bound = maxVal + voxel_size3 * 0.5;
    Eigen::Vector3d voxel_min_bound = minVal - voxel_size3 * 0.5;

    if (voxel_size * std::numeric_limits<int>::max() <
        (voxel_max_bound - voxel_min_bound).maxCoeff()) {
        std::cout << "[VoxelDownSample] voxel_size is too small." << std::endl;
    }
    std::unordered_map<Eigen::Vector3i, AccumulatedPoint, hash_eigen<Eigen::Vector3i>> voxelindex_to_accpoint;

    Eigen::Vector3d ref_coord;
    Eigen::Vector3i voxel_index;
    size_t nr_new_points = 0;
    for (int i = 0; i < (int)points.rows(); i++) {
        Eigen::Vector3d p = points.row(i);
        ref_coord = (p - voxel_min_bound) / voxel_size;
        voxel_index << int(floor(ref_coord(0))), int(floor(ref_coord(1))),
            int(floor(ref_coord(2)));
        voxelindex_to_accpoint[voxel_index].AddPoint(p);
        nr_new_points++;
    }

    Eigen::MatrixXd output(voxelindex_to_accpoint.size(), 3);

    size_t counter = 0;
    for (auto accpoint : voxelindex_to_accpoint) {
        Eigen::Vector3d new_p = accpoint.second.GetMedianPoint();

        output.row(counter) << new_p(0), new_p(1), new_p(2);
        counter++;
    }
    return output;
}

Eigen::MatrixXd VoxelDownSampleColor(Eigen::Matrix<double, Eigen::Dynamic, 3> &points,
                                     Eigen::Matrix<double, Eigen::Dynamic, 3> &feats,
                                     double voxel_size) {
    if (voxel_size <= 0.0) {
        std::cout << "[VoxelDownSample] voxel_size <= 0." << std::endl;
    }

    Eigen::Vector3d voxel_size3 = Eigen::Vector3d(voxel_size, voxel_size, voxel_size);
    Eigen::Vector3d maxVal = points.colwise().maxCoeff();
    Eigen::Vector3d minVal = points.colwise().minCoeff();

    Eigen::Vector3d voxel_max_bound = maxVal + voxel_size3 * 0.5;
    Eigen::Vector3d voxel_min_bound = minVal - voxel_size3 * 0.5;

    if (voxel_size * std::numeric_limits<int>::max() <
        (voxel_max_bound - voxel_min_bound).maxCoeff()) {
        std::cout << "[VoxelDownSample] voxel_size is too small." << std::endl;
    }
    std::unordered_map<Eigen::Vector3i, AccumulatedPoint, hash_eigen<Eigen::Vector3i>> voxelindex_to_accpoint;

    Eigen::Vector3d ref_coord;
    Eigen::Vector3i voxel_index;
    for (int i = 0; i < (int)points.rows(); i++) {
        Eigen::Vector3d p = points.row(i);
        Eigen::Vector3d c = feats.row(i);
        float ref_coord_x = (p(0) - voxel_min_bound(0)) / voxel_size;
        float ref_coord_y = (p(1) - voxel_min_bound(1)) / voxel_size;
        float ref_coord_z = (p(2) - voxel_min_bound(2)) / voxel_size;
        voxel_index << int(floor(ref_coord_x)), int(floor(ref_coord_y)),
            int(floor(ref_coord_z));
        voxelindex_to_accpoint[voxel_index].AddPoint(p, c);
    }

    Eigen::MatrixXd output(voxelindex_to_accpoint.size(), 6);

    size_t counter = 0;
    int count_often = 0;
    for (auto accpoint : voxelindex_to_accpoint) {
        Eigen::Vector3d new_p = accpoint.second.GetAveragePoint();
        Eigen::Vector3d new_c = accpoint.second.GetAverageColor();

        // accpoint.second.CalcCovarianceMatrix();
        // Eigen::Vector3d eigen_vals = accpoint.second.GetEigenValues();
        output.row(counter) << new_p(0), new_p(1), new_p(2),
            new_c(0), new_c(1), new_c(2);
        counter++;
    }
    return output;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> VoxelDownSampleLabel(Eigen::Matrix<double, Eigen::Dynamic, 3> points,
                                                                  Eigen::Matrix<double, Eigen::Dynamic, 3> feats,
                                                                  Eigen::VectorXd label,
                                                                  double voxel_size) {
    if (voxel_size <= 0.0) {
        std::cout << "[VoxelDownSample] voxel_size <= 0." << std::endl;
    }
    Eigen::Vector3d voxel_size3 = Eigen::Vector3d(voxel_size, voxel_size, voxel_size);
    Eigen::Vector3d maxVal = points.colwise().maxCoeff();
    Eigen::Vector3d minVal = points.colwise().minCoeff();
    Eigen::Vector3d voxel_max_bound = maxVal + voxel_size3 * 0.5;
    Eigen::Vector3d voxel_min_bound = minVal - voxel_size3 * 0.5;

    if (voxel_size * std::numeric_limits<int>::max() <
        (voxel_max_bound - voxel_min_bound).maxCoeff()) {
        std::cout << "[VoxelDownSample] voxel_size is too small." << std::endl;
    }
    std::unordered_map<Eigen::Vector3i, AccumulatedPointLabel, hash_eigen<Eigen::Vector3i>> voxelindex_to_accpoint;
    Eigen::MatrixXi voxel_indizies(points.rows(), 3);

    Eigen::Vector3d ref_coord;
    Eigen::Vector3i voxel_index;
    size_t nr_new_points = 0;
    for (int i = 0; i < (int)points.rows(); i++) {
        Eigen::Vector3d p = points.row(i);
        Eigen::Vector3d c = feats.row(i);
        int l = label[i];
        ref_coord = (p - voxel_min_bound) / voxel_size;
        voxel_index << int(floor(ref_coord(0))), int(floor(ref_coord(1))),
            int(floor(ref_coord(2)));
        voxelindex_to_accpoint[voxel_index].AddPoint(p, c, l);
        nr_new_points++;
        voxel_indizies.row(i) << voxel_index(0), voxel_index(1), voxel_index(2);
    }

    Eigen::MatrixXd output(voxelindex_to_accpoint.size(), 7);

    size_t counter = 0;
    for (auto accpoint : voxelindex_to_accpoint) {
        Eigen::Vector3d new_p = accpoint.second.GetAveragePoint();
        Eigen::Vector3d new_c = accpoint.second.GetAverageColor();

        int new_label = accpoint.second.VoteLabel();

        output.row(counter) << new_p(0), new_p(1), new_p(2),
            new_label,
            new_c(0), new_c(1), new_c(2);
        counter++;
    }
    return std::make_tuple(output, voxel_indizies.cast<double>());
}

PYBIND11_MODULE(downsampling, m) {
    m.doc() = "pybind11 plugin for voxel downsampling";  // optional module docstring
    m.def("voxel_down", &VoxelDownSample);
    m.def("voxel_down_color", &VoxelDownSampleColor);
    m.def("voxel_down_label", &VoxelDownSampleLabel);
}