
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
        cout << "ERROR: Cannot find dataset." << endl;
        return 1;
    }

    vector<Gesture> dataset;
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        Gesture gesture;
        string value;

        getline(ss, gesture.label, ',');

        while (getline(ss, value, ',')) {
            try {
                gesture.features.push_back(stod(value));
            } catch (...) {
                break;
            }
        }

        if (
            gesture.features.size() == 42 &&
            (gesture.label == "OPEN" ||
             gesture.label == "FIST" ||
             gesture.label == "PEACE")
        ) {
            dataset.push_back(gesture);
        }
    }

    if (dataset.size() < 4) {
        cout << "Not enough examples to test." << endl;
        return 1;
    }

    if (dataset.size() < 2) {
        cout << "Not enough examples to test." << endl;
        return 1;
    }
map<string, int> gesture_total;
map<string, int> gesture_correct;
    int correct = 0;
    int total = 0;

    for (int i = 0; i < (int)dataset.size(); i++) {
        vector<pair<double, string>> neighbors;

        for (int j = 0; j < (int)dataset.size(); j++) {
            // Never compare an example with itself
            if (i == j) {
                continue;
            }

            double distance = distanceBetween(
                dataset[i].features,
                dataset[j].features
            );

            neighbors.push_back({
                distance,
                dataset[j].label
            });
        }

        sort(neighbors.begin(), neighbors.end());

        map<string, int> votes;

        int k = min(3, (int)neighbors.size());

        for (int neighbor = 0; neighbor < k; neighbor++) {
            votes[neighbors[neighbor].second]++;
        }

        string prediction = neighbors[0].second;
        int highestVotes = votes[prediction];

        for (const auto& vote : votes) {
            if (vote.second > highestVotes) {
                prediction = vote.first;
                highestVotes = vote.second;
            }
        }

        
gesture_total[dataset[i].label]++;

if (prediction == dataset[i].label) {
    correct++;
    gesture_correct[dataset[i].label]++;
}

total++;
    }

    double accuracy = 100.0 * correct / total;

    cout << "Total examples: " << total << endl;
    cout << "Correct predictions: " << correct << endl;
    cout << "Accuracy: " << accuracy << "%" << endl;

cout << "\nAccuracy by gesture:" << endl;

for (const string& label : {"OPEN", "FIST", "PEACE"}) {
    int tested = gesture_total[label];
    int passed = gesture_correct[label];

    if (tested > 0) {
        double percentage = 100.0 * passed / tested;

        cout << label << ": "
             << passed << "/" << tested
             << " (" << percentage << "%)" << endl;
    }
}


    return 0;
}
