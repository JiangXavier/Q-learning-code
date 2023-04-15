//
// Created by retell123 on 2023/4/15.
//
#include <iostream>
#include <fstream>
#include<cmath>
using namespace std;
const int MAXN=10000000,MAXM=1000;
int x[MAXN],y[MAXN],mapp[MAXM][MAXM],cons[MAXM][MAXM],bb[MAXN];
int cnt;
int f=0;
int n;
bool vis[MAXN];
int stack[MAXN],sta=0;
double dist[MAXM][MAXM];
double cal(int a,int b)
{
    double xa=x[a],xb=x[b],ya=y[a],yb=y[b];
    return sqrt((xa-xb)*(xa-xb)+(ya-yb)*(ya-yb));
}
void search(int xx,int steps,int len,int head)
{
    if(steps==1)
    {
        f++;
        for(int i=1;i<=sta;i++)
            cons[f][stack[i]]=1;
        cons[f][mapp[xx][head]]=1;
        bb[f]=len-1;
        return;
    }
    for(int i=1;i<=n;i++)
    {
        if(!vis[i]&&mapp[xx][i])
        {
            vis[i]=1;
            sta++;
            stack[sta]=mapp[xx][i];
            search(i,steps-1,len,head);
            sta--;
            vis[i]=0;
        }
    }
}
int main()
{
    freopen("o.txt","w",stdout);
    cin>>n;
    for(int i=1;i<=n;i++) {
        int aa;
        cin >> aa >> x[i] >> y[i];
    }
    for(int i=1;i<=n;i++)
        for(int j=1;j<=n;j++)
        {
            if(i==j)
            {
                dist[i][j]=0;
            }
            else
            {
                cnt++;
                mapp[i][j]=cnt;
                dist[i][j]=cal(i,j);
            }
        }


    //ºá
    for(int i=1;i<=n;i++)
    {
        f++;
        for(int j=1;j<=n;j++)
        {
            if(mapp[i][j]!=0)
                cons[f][mapp[i][j]]=1;
        }
        bb[f]=1;
        f++;
        for(int j=1;j<=n;j++)
        {
            if(mapp[i][j]!=0)
                cons[f][mapp[i][j]]=-1;
        }
        bb[f]=-1;

    }
    //Êú
    for(int i=1;i<=n;i++)
    {
        f++;
        for(int j=1;j<=n;j++)
        {
            if(mapp[j][i]!=0)
                cons[f][mapp[j][i]]=1;
        }
        bb[f]=1;
        f++;
        for(int j=1;j<=n;j++)
        {
            if(mapp[j][i]!=0)
                cons[f][mapp[j][i]]=-1;
        }
        bb[f]=-1;
    }

    for(int i=2;i<n;i++)
    {
        for(int j=1;j<=n;j++)
        {
            vis[j]=1;
            search(j,i,i,j);
            vis[j]=0;

        }
    }
//    for(int i=1;i<=n;i++) {
//
//        for (int j = 1; j <= n; j++)
//            printf("%d ",mapp[i][j]);
//        printf("\n");
//    }


    printf("%d\n",cnt);
    for(int i=1;i<=n;i++)
        for(int j=1;j<=n;j++)
            if(mapp[i][j])
                cout<<-dist[i][j]<<" ";
    printf("\n");

    for(int i=1;i<=f;i++)
    {
        for(int j=1;j<=cnt;j++)
            printf("%d ",cons[i][j]);
        printf("%d\n",bb[i]);
    }
}