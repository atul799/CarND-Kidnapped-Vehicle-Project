#include <iostream>
#include <map>
#include <random>
 
int main()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    //std::discrete_distribution<> d({40, 10, 10, 40});
    std::vector<double> weights={0.40, 0.10, 0.20, 0.10};
    std::discrete_distribution<> d(weights.begin(),weights.end());
    //std::discrete_distribution<> d({0.40, 0.10, 0.20, 0.10});
    //std::map<int, int> m;

    for(int n=0; n<10; ++n) {
        //++m[d(gen)];
        std::cout << d(gen)<<"\n";
    }
    //for(auto p : m) {
    //    std::cout << p.first << " generated " << p.second << " times\n";
    //}
}