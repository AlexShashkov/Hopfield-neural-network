#include "neural.h"

using namespace std;

int main(){
    Neural::Network net("example.txt");
    net.print();

    vector<float> test = {1, -1, -1, 1, -1, -1, 1, 1, -1};
    vector<float> result = net.eval(test);
    cout << "Исходная: ";
    Neural::print_arr(test);
    cout << "Искомая:  1 -1 -1 1 -1 -1 1 1 1\nРезультат:";
    Neural::print_arr(result);

    cout << "Тест на помехи\n";
    Neural::Network net2("example2.txt", Neural::starToState);
    // net.print(); Матрица 100x100 слишком большая, чтобы ее показать
    int max_sim_v = 0;
    std::vector<float> last_eval_v;
    std::vector<float> filtered_v = net2("noisy_k.txt", Neural::starToState, 1000, &max_sim_v, &last_eval_v);
    cout << "ОТФИЛЬТРОВАНО\n";
    cout << "Отфильтрованный образ: "; Neural::print_arr(filtered_v);
    cout << "max совпадение: " << max_sim_v;
    cout << "\nПоследнее исполнение: "; Neural::print_arr(last_eval_v);

    int cntr = 0;
    cout << "\n";
    for_each(filtered_v.begin(), filtered_v.end(),
        [&cntr](auto val) {
            cout << val << " ";
            if(cntr++ == 9){ cout << "\n"; cntr = 0;}
        }
    );
}