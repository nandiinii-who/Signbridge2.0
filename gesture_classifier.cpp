
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

// Calculate similarity using Euclidean distance
double distanceBetween(
    const vector<double>& a,
    const vector<double>& b
) {
    double sum = 0;

    for (int i = 0; i < 42; i++) {
        double difference = a[i] - b[i];
        sum += difference * difference;
    }

    return sqrt(sum);
}

int main() {
    ifstream file("data/gestures.csv");

    if (!file.is_open()) {
        cout << "ERROR: Cannot find gestures.csv" << endl;
        return 1;
    }

    vector<Gesture> dataset;
    string line;

    // Read saved gesture examples
    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        Gesture gesture;

        getline(ss, gesture.label, ',');

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

    file.close();

    cout << "Loaded " << dataset.size()
         << " gesture examples." << endl;

    if (dataset.size() < 2) {
        cout << "Not enough training examples." << endl;
        return 1;
    }

    // Test the classifier on the last saved example
    Gesture test = dataset.back();
    dataset.pop_back();

    vector<pair<double, string>> neighbors;

    for (const Gesture& example : dataset) {
        double distance = distanceBetween(
            test.features,
            example.features
        );

        neighbors.push_back({
            distance, example.label
        });
    }

    // Sort by distance: closest examples first
    sort(neighbors.begin(), neighbors.end());

    // KNN with up to 3 nearest neighbors
    int k = min(3, (int)neighbors.size());

    map<string, int> votes;

    for (int i = 0; i < k; i++) {
        votes[neighbors[i].second]++;
    }

    string prediction = neighbors[0].second;
    int highestVotes = votes[prediction];

    for (const auto& vote : votes) {
        if (vote.second > highestVotes) {
            prediction = vote.first;
            highestVotes = vote.second;
        }
    }

    cout << "Actual gesture: "
         << test.label << endl;

    cout << "Predicted gesture: "
         << prediction << endl;

    if (prediction == test.label) {
        cout << "Prediction correct!" << endl;
    } else {
        cout << "Prediction incorrect." << endl;
    }

    return 0;
}
