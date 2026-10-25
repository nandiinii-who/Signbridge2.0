
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <map>

using namespace std;

struct Gesture {
    string label;
    vector<double> features;
};

double distanceBetween(
    const vector<double>& a,
    const vector<double>& b
) {
    double sum = 0;

    for (int i = 0; i < 42; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sqrt(sum);
}

int main() {
    ifstream file("data/gestures.csv");

    if (!file.is_open()) {
        cerr << "ERROR: Dataset not found." << endl;
        return 1;
    }

    vector<Gesture> dataset;
    string line;

    // Read training examples
    while (getline(file, line)) {
        stringstream ss(line);
        Gesture gesture;
        string value;

        getline(ss, gesture.label, ',');

        if (
            gesture.label != "OPEN" &&
            gesture.label != "FIST" &&
            gesture.label != "PEACE"
        ) {
            continue;
        }

        while (getline(ss, value, ',')) {
            try {
                gesture.features.push_back(stod(value));
            } catch (...) {
                break;
            }
        }

        if (gesture.features.size() == 42) {
            dataset.push_back(gesture);
        }
    }

    if (dataset.empty()) {
        cerr << "ERROR: No training examples." << endl;
        return 1;
    }

    // Receive 42 landmark values from Python
    vector<double> input(42);

    for (int i = 0; i < 42; i++) {
        if (!(cin >> input[i])) {
            cerr << "ERROR: Expected 42 values." << endl;
            return 1;
        }
    }

    // Calculate distance to every saved example
    vector<pair<double, string>> neighbors;

    for (const Gesture& example : dataset) {
        neighbors.push_back({
            distanceBetween(input, example.features),
            example.label
        });
    }

    // Sort from closest to farthest
    sort(neighbors.begin(), neighbors.end());

    // Find the 3 nearest neighbors
    int k = min(3, (int)neighbors.size());

    map<string, int> votes;

    for (int i = 0; i < k; i++) {
        votes[neighbors[i].second]++;
    }

    // Choose the gesture with the most votes
    string prediction = "";
    int highestVotes = -1;

    for (const auto& item : votes) {
        if (item.second > highestVotes) {
            highestVotes = item.second;
            prediction = item.first;
        }
    }

    cout << prediction << endl;

    return 0;
}
