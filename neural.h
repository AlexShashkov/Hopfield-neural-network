#ifndef NEURAL_H
#define NEURAL_H

#include <algorithm>
#include <iterator>
#include <iostream>
#include <numeric>
#include <fstream>
#include <vector>

namespace Neural
{
    typedef std::vector<std::vector<float>> Matrix;
    enum states{ACTIVE=1, INACTIVE=-1};

    auto dot = [](std::vector<float> vec, float val, std::vector<float>& newvec){
        std::for_each(vec.begin(), vec.end(), 
            [&newvec, val](float val2){newvec.push_back(val*val2);}
        );
    };

    auto print_arr = [](auto arr){
        std::for_each(
            arr.begin(), arr.end(),
            [](auto i) {std::cout << i << " ";}
        );
        std::cout << "\n";
    };

    int starToState(std::ifstream& stream){
        char symbol; stream >> symbol;
        if(symbol == '*') return 1;
        else return -1;
    }


    class Neuron{
        private:
            std::vector<float> weights;
        public:
            Neuron(std::vector<float> val){
                weights = val;
            }
            Neuron(int count){
                weights = std::vector<float>(count, 0.0);;
            }

            std::vector<float> get(){
                return weights;
            };
            void clear(){
                weights.clear();
            };

            int operator() (std::vector<float> input){
                int ctr = 0;
                float res =  std::accumulate(
                    weights.begin(), weights.end(), 0.0,
                    [&ctr, &input](float sum, float weight){ return sum + (weight*input[ctr++]); }
                );

                return res > 0 ? ACTIVE : INACTIVE; 
            }

            void print(){
                print_arr(weights);
            }

        friend std::ofstream& operator<<(std::ofstream&, Neuron&);
        friend std::ifstream& operator>>(std::ifstream&, Neuron&);
    };


    std::ofstream& operator<<(std::ofstream& stream, Neuron& neuron){
        std::for_each(
            neuron.weights.begin(), neuron.weights.end(),
            [&stream](auto i) {stream << i << " ";}
        );
        stream << "\n";
    }

    std::ifstream& operator>>(std::ifstream& stream, Neuron& neuron){
        std::transform(
            neuron.weights.begin(), neuron.weights.end(), neuron.weights.begin(),
            [&stream](auto i) {
                stream >> i;
                return i;
            }
        );
    }


    class Network{
        private:
            int image=0, size=0;
            std::vector<Neuron> neurons;
            Matrix keys;

            std::vector<float> read_image(std::ifstream& stream, int (*lambda)(std::ifstream& strm)=nullptr){
                std::vector<float> images;
                for(int i=0; i<size*size; i++){
                        int num;
                        if(lambda != nullptr) num = lambda(stream);
                        else stream >> num;
                        images.push_back(num);
                }
                return images;
            }

            Matrix train_image(Matrix matrix, std::vector<float> img){
                std::for_each(img.begin(), img.end(),
                    [&matrix, img](float val){
                        std::vector<float> newvec;
                        dot(img, val, newvec);
                        matrix.push_back(newvec);
                    }
                );

                return matrix;
            }
        public:

            /**
                Создание и обучение нейронной сети Хопфилда, используя образы из файла.
                @param file the single digit to encode.
                @param lambda=nullptr функция, используемая для обработки символов из текста, принимающая на вход поток ifstream. По умолчанию будет читать текст напрямую
            */
            Network(std::string file, int (*lambda)(std::ifstream& stream)=nullptr){
                train(file, lambda);
            };

            /**
                Создать объект нейронной сети.
            */
            Network(){};

            /**
                Обучение нейронной сети Хопфилда, используя образы из файла.
                @param file the single digit to encode.
                @param lambda=nullptr функция, используемая для обработки символов из текста. По умолчанию будет читать текст напрямую
            */
            void train(std::string file, int (*lambda)(std::ifstream& strm)=nullptr){
                std::ifstream stream(file);
                stream >> image >> size;

                std::cout << "Обучается " << image << " образа размером " << size << "x" << size << "\n";

                Matrix matrix;
                for(int i = 0; i < size*size; i++){
                    matrix.push_back(std::vector<float>(size*size, 0.0));
                }

                for(int i=0; i<image; i++){
                    std::vector<float> img = read_image(stream, lambda);
                    std::cout << "Образ №" << i << "\n";
                    print_arr(img);

                    keys.push_back(img);

                    Matrix trained;
                    trained = train_image(trained, img);
                    for(int j=0; j<matrix.size(); j++){
                        std::transform(trained[j].begin(), trained[j].end(), matrix[j].begin(), matrix[j].begin(), std::plus<int>());
                    }
                }

                std::cout << "Count of neurons: " << size*size << "\n";

                float neuordiv = 1/((float)size*(float)size);

                // std::cout << "Divider: " << neuordiv << "\n";

                for(int i=0; i<matrix.size(); i++){
                    std::transform(matrix[i].begin(), matrix[i].end(), matrix[i].begin(),
                        [neuordiv](auto val) -> auto { 
                            return val*neuordiv;
                        }
                    );
                    matrix[i][i] = 0;
                    neurons.push_back(Neuron(matrix[i]));
                    // std::cout << "Row #" << i << " transformed\n";
                }
                stream.close();
                std::cout << "Нейронная сеть обучена.\n";
            }

            /**
                Вывести весовую матрицу нейронной сети Хопфилда.
            */
            void print(){
                std::cout << "Trained net:\n";
                std::for_each(neurons.begin(), neurons.end(),
                    [](Neuron n) {n.print();}
                );
                std::cout << "Images the net was trained on:\n";
                std::for_each(keys.begin(), keys.end(), print_arr);
            }

            /**
                Один асинхронный проход нейронной сети.
                @param input Вектор входных данных типа float
                @returns Результирующий вектор.
            */
            std::vector<float> eval(std::vector<float> input){
                std::transform(neurons.begin(), neurons.end(), input.begin(),
                    [input](Neuron n) -> float {
                        return n(input);
                    }
                );

                return input;
            }

            /**
                Фильтрация образа 
                @param input Вектор входных данных типа float
                @param max_cntr=1000 Максимальное количество итераций.
                @param max_sim=nullptr Если указан - запишет как много нейронов совпало.
                @param last_eval=nullptr Если указан - запишет последний обработанный вектор.
                @returns Результирующий вектор.
            */
            std::vector<float> operator() (std::vector<float> input, int max_cntr=1000, int* max_sim=nullptr, std::vector<float>* last_eval=nullptr){
                int max_simmilarity = -1;
                int cntr = 0;
                std::vector<float> most_simmiliar;

                while(cntr < max_cntr && max_simmilarity != size*size){
                    input = eval(input);
                    std::for_each(keys.begin(), keys.end(), 
                        [&max_simmilarity, &most_simmiliar, input](auto vec){
                            int n = inner_product(vec.begin(), vec.end(), input.begin(), 0, std::plus<>(), std::equal_to<>());
                            if(n > max_simmilarity){ most_simmiliar = vec; max_simmilarity=n; }
                        }
                    );
                    if(cntr % 100 == 0){
                        std::cout << "Текущее состояние фильтрации:\n Проход #" << cntr <<
                            ", наибольшая схожесть: " << max_simmilarity << " из " << size*size << "\n";
                        //std::cout << "Больше всего похоже на: "; print_arr(most_simmiliar);
                        //std::cout << "Последнее исполнение: "; print_arr(input);
                    }
                    cntr++;
                }
                if(max_sim != nullptr) *max_sim = max_simmilarity;
                if(last_eval != nullptr) *last_eval = input;
                return most_simmiliar;
            }

            /**
                Фильтрация образа через текстовый файл
                @param input Вектор входных данных типа float
                @param max_cntr=1000 Максимальное количество итераций.
                @param max_sim=nullptr Если указан - запишет как много нейронов совпало.
                @param last_eval=nullptr Если указан - запишет последний обработанный вектор.
                @returns Результирующий вектор.
            */
            std::vector<float> operator() (std::string file, int (*lambda)(std::ifstream& strm)=nullptr, int max_cntr=1000,
                                                             int* max_sim=nullptr, std::vector<float>* last_eval=nullptr ){
                std::ifstream stream(file);
                std::vector<float> img = read_image(stream, lambda);
                stream.close();
                std::cout << "Прочитан образ: "; print_arr(img); 
                return operator()(img, max_cntr, max_sim, last_eval);
            }

        friend std::ofstream& operator<<(std::ofstream&, Network&);
        friend std::ifstream& operator>>(std::ifstream&, Network&);
    };


    std::ofstream& operator<<(std::ofstream& stream, Network& network){
        stream << network.image << " " << network.size << "\n";
        std::for_each(
            network.neurons.begin(), network.neurons.end(),
                [&stream](Neuron n) {
                    stream << n;
                }
        );

        std::for_each(
            network.keys.begin(), network.keys.end(),
                [&stream](auto vec) {
                    for (auto &item : vec) stream << item << " ";
                    stream << "\n";
                }
        );
    }

    std::ifstream& operator>>(std::ifstream& stream, Network& network){
        network.neurons.clear();
        network.keys.clear();
        stream >> network.image >> network.size;

        for(int i = 0; i < network.size*network.size; i++){
            Neuron neuron(network.size*network.size);
            stream >> neuron;
            network.neurons.push_back(neuron);
        }
        for(int i = 0; i < network.image; i++){
            std::vector<float> vec;
            for(int i = 0; i < network.size*network.size; i++){
                float ret;
                stream >> ret;
                vec.push_back(ret);
            }
            network.keys.push_back(vec);
        }
    }
}

#endif