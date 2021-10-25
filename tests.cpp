#include <iostream>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <vector>

using namespace std;



int main(){

    // Произведение строки на число
    vector<float> weights = {
        1.5, 1.1, 2, 3, 5
    };

    vector<float> weights2 = {
        1, 0, 1, 1, 1
    };

    int val = 1;

    float res =  std::accumulate(
        weights.begin(), weights.end(),
        0.0,
        [&val](float sum, float weight){ return sum + (weight * val); }
    );

    // Произведение строки на столбец

    int ctr = 0;

    res =  std::accumulate(
        weights.begin(), weights.end(),
        0.0,
        [&ctr, &weights2](float sum, float weight){ return sum + (weight*weights2[ctr++]); }
    );

    cout << res << endl;

    // Чтение из файла

    int image, size;
    std::ifstream stream("example.txt");
    stream >> image >> size;

    std::vector<std::vector<double>> matrix;
    for(int i = 0; i < size; i++){
        matrix.push_back(std::vector<double>(size*size, 0.0));
    }
    
    std::vector<std::vector<double>> images;
    for(int i=0; i<image; i++){
        std::vector<double> row;
        for(int j = 0; j<size; j++){
            int num;
            stream >> num;
            row.push_back(num);
        };
        images.push_back(row);
    }
    stream.close();

    auto print_arr = [](auto arr){
        for_each(
            arr.begin(), arr.end(),
            [](double i) {std::cout << i << " ";}
        );
        cout << "\n";
    };

    cout << "-----------------\n";
    for_each(matrix.begin(), matrix.end(), print_arr);
    cout << "-----------------\n";

    cout << "\n-----------------\n";
    for_each(images.begin(), images.end(), print_arr);
    cout << "-----------------\n";

    // Обучение

    vector<float> test1 = {
        1, -2, 3, 4, 5
    };

    vector<vector<float>> test2;
    vector<vector<float>> train;
    for(int i=0; i<5; i++)
        train.push_back(vector<float>(5, 0.0));

    auto dot = [](std::vector<float> vec, float val, std::vector<float>& newvec){
        std::for_each(vec.begin(), vec.end(), 
            [&newvec, val](float val2){newvec.push_back(val*val2);}
        );
    };

    std::for_each(test1.begin(), test1.end(),
        [&test2, test1, dot](float val) {
            std::vector<float> newvec;
            dot(test1, val, newvec);
            test2.push_back(newvec);
        }
    );

    cout << "before\n";

    for_each(train.begin(), train.end(), print_arr);

    for(int i=0; i<train.size(); i++){
        std::transform(test2[i].begin(), test2[i].end(), test2[i].begin(), train[i].begin(), std::plus<int>());
    }

    cout << "after\n";

    for_each(train.begin(), train.end(), print_arr);

    cout << "*2\n";

    for(int i=0; i<train.size(); i++){
        std::transform(train[i].begin(), train[i].end(), train[i].begin(), train[i].begin(), std::plus<int>());
    }

    for_each(train.begin(), train.end(), print_arr);

    cout << "умножение на число\n";

    for(int i=0; i<train.size(); i++){
        std::transform(train[i].begin(), train[i].end(), train[i].begin(),
                   [](auto val) -> auto { return val*0.1; });
    }

    for_each(train.begin(), train.end(), print_arr);
}

