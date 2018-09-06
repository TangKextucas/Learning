#include<stdio.h>
void swap(int v[],int i,int j)
{
    int temp;
    temp=v[i];
    v[i]=v[j];
    v[j]=temp;
}
void qsort(int v[],int left,int right)
{
    if(left>=right)
        return;
    swap(v,left,(left+right)/2);
    int last=left;
    for(int i=left+1;i<=right;i++)
        if(v[i]<v[left])
            swap(v,++last,i);
    swap(v,left,last);
    qsort(v,left,last-1);
    qsort(v,last+1,right);
}
int main(void)
{
    int gg[]={4,6,5,2,7,3,9,8,0,1,6,23,7,89,478,523,-100,-34,-6,-3,-1,19,16,13,15};
    int right=sizeof(gg)/sizeof(gg[0]);
    qsort(gg,0,right-1);
    for(int i=0;i<right;i++)
        printf("%d ",gg[i]);
    return 0;  
}
