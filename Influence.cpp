#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include "sha3.h"

using namespace std;


typedef unsigned long long ull;



/*
    候选变量位置

    I0中6个黄色lane:

    A[0,0]
    A[2,0]

    A[0,1]
    A[2,1]

    A[0,2]
    A[2,2]

*/

struct Variable
{
    int x;
    int y;
    int z;
};



vector<Variable> candidate;



/*
    初始化384个候选变量

    每个lane:
        64 bit

    共:
        6*64=384

*/

void InitVariables()
{

    int yellow_lane[6][2]
        =
    {
        {0,0},
        {2,0},

        {0,1},
        {2,1},

        {0,2},
        {2,2}
    };


    for (int i = 0; i < 6; i++)
    {

        int x =
            yellow_lane[i][0];

        int y =
            yellow_lane[i][1];


        for (int z = 0; z < 64; z++)
        {

            candidate.push_back(
                { x,y,z }
            );

        }

    }

}



/*
    设置一个候选变量

    Keccak状态:

        state[x+5*y]=A[x,y]

*/

void SetVariable(
    ull state[25],
    Variable v
)
{

    int lane =
        v.x + 5 * v.y;


    state[lane]
        ^=
        (1ULL << v.z);

}



/*
    三轮攻击相关传播


    Round 1:
        theta
        rho
        pi
        chi
        iota


    Round 2:
        theta
        rho
        pi
        chi
        iota


    Round 3:
        theta
        rho
        pi


    停止在最后chi之前

    即得到 Π3

*/

void Keccak3_Attack(
    ull state[25]
)
{

    // Round 1

    theta(state);
    rho_pi(state);
    chi(state);
    iota(state, 0);



    // Round 2

    theta(state);
    rho_pi(state);
    chi(state);
    iota(state, 1);



    // Round 3

    // 不执行chi和iota

    theta(state);
    rho_pi(state);

}



/*
    统计攻击相关plane


    Π3中:

        y=0

        y=1


    共:

        5*2*64

        =640 bit


*/

int AttackInfluence(
    ull state[25]
)
{

    int influence = 0;



    for (int y = 0; y <= 1; y++)
    {

        for (int x = 0; x < 5; x++)
        {

            int lane =
                x + 5 * y;


            ull value =
                state[lane];



            while (value)
            {

                influence +=
                    value & 1ULL;


                value >>= 1;

            }

        }

    }


    return influence;

}




int main()
{


    cout
        << "============================"
        << endl;


    cout
        << "3-round Keccak-384"
        << endl;


    cout
        << "Attack Influence Test"
        << endl;


    cout
        << "============================"
        << endl;



    InitVariables();



    cout
        << "Number of candidate variables: "
        << candidate.size()
        << endl;



    vector<pair<int, int>> result;



    /*
        逐个测试384个候选变量

    */

    for (int i = 0; i < (int)candidate.size(); i++)
    {


        // 初始状态 I0=0

        ull state[25] = { 0 };



        // 设置xi=1

        SetVariable(
            state,
            candidate[i]
        );



        // 三轮传播到 Π3

        Keccak3_Attack(state);



        // 计算Influence

        int value =
            AttackInfluence(state);



        result.push_back(
            { i,value }
        );



        cout
            << "variable "
            << i
            << "  Influence = "
            << value
            << endl;


    }




    /*
        保存原始结果

    */

    ofstream fout(
        "attack_influence.csv"
    );


    fout
        << "variable,Influence\n";



    for (auto& item : result)
    {

        fout
            << item.first
            << ","
            << item.second
            << "\n";

    }


    fout.close();




    /*
        按Influence排序

    */

    sort(
        result.begin(),
        result.end(),
        [](auto& a, auto& b)
        {

            return a.second > b.second;

        }
    );



    ofstream fout2(
        "attack_influence_rank.csv"
    );


    fout2
        << "rank,variable,Influence\n";


    int rank = 1;


    for (auto& item : result)
    {

        fout2
            << rank++
            << ","
            << item.first
            << ","
            << item.second
            << "\n";

    }


    fout2.close();




    cout << endl;

    cout
        << "Finished!"
        << endl;


    cout
        << "Generated files:"
        << endl;


    cout
        << "attack_influence.csv"
        << endl;


    cout
        << "attack_influence_rank.csv"
        << endl;



    return 0;

}