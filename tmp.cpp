
#include <random>

using namespace std;


random_device rd;
default_random_engine gen;



int main() {



    
    
    discrete_distribution<> d({.5, 0.10, 0.10, 0.3});
    //std::map<int, int> m;
    for(int n=0; n<10000; ++n) {
        //++m[d(gen)];
        cout << d(gen) <<"\n";
    }
    //for(auto p : m) {
    //    std::cout << p.first << " generated " << p.second << " times\n";
    //}
return 0;

}
