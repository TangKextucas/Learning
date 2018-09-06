#include<stdio.h>
#include<math.h>
#include<stdbool.h>
#include<stdint.h>
#include<stdlib.h>
typedef int64_t GreatNum;
bool IsPrime(GreatNum n){
    if(n<=1)
        return false;
    GreatNum bound=(GreatNum)sqrt(n)+1;
    for(GreatNum i=2;i<bound;i++)
        if(n%i==0)
            return false;
    return true;
}
void PrintAllPrimeLessThanN(GreatNum n){
    bool *state=(bool*)malloc(sizeof(bool)*(n+1));
    for(GreatNum i=2;i<=n;i++)		
        if(!state[i]){
            printf("  %lld",i);
            for(GreatNum j=i*i;j<=n;j+=i)
                state[j]=true;
        }
    free(state);
}
int main(void){
    printf("%d\n",IsPrime(19260817));
    PrintAllPrimeLessThanN(100);
}
