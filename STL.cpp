#include<cstdio>
#include<vector>                           //向量(变长数组)
#include<set>      //集合(内部自动有序，不含重复元素的容器，红黑树实现)
#include<string>                        //字符串类
#include<map>           //建立两个基本类型之间的映射，红黑树实现
#include<queue>             //队列和优先队列(堆实现)
#include<stack>             //栈
#include<utility>
#include<algorithm>             
using namespace std;
struct fruit                //水果结构体
{
    string name;
    int price;
    friend bool operator < (fruit f1,fruit f2)  //定义一个友元函数
    {                                    //C++不允许重载大于运算符
        return f1.price > f2.price;
    }       //把水果放进优先队列后，堆顶元素会是price最小的那一个水果
}f1,f2,f3;      
bool cmp(string str1,string str2)       //用于sort的比较函数
{
    return str1.length()<str2.length();     //按长度从小到大排序
}
int main()
{
//vector
    vector<int> vi;                             //定义
    for(int i=0;i<=8;i++)
        vi.push_back(i);                //添加尾部元素
for(vector<int>::iterator it=vi.begin();it!=vi.end();it++)  
										//迭代器访问
        printf("%2d",*it);              //迭代器是一个类似指针的东西
    printf("\n%2d\n",vi.size());                    //打印元素个数
    vi.insert(vi.begin()+2,-1);                 //将-1插入vi[2]的位置
    vi.pop_back();                              //删除尾元素
    vi.erase(vi.begin()+4);                     //删除vi[4]位置的元素
    vi.erase(vi.begin()+5,vi.begin()+7);   //删除vi[5]到vi[7-1]位置的元素
    for(int i=0;i<vi.size();i++)
        printf("%2d",vi[i]);                //下标访问
    vi.clear();                 //清空其中所有元素
    printf("\n");
//set
    set<int> st;
    for(int i=-2;i<=8;i+=2)
        st.insert(i);               //插入元素
    for(set<int>::iterator it=st.begin();it!=st.end();it++) //迭代器访问
        printf("%2d",*it);
    printf("\n%2d\n",st.size());
    st.erase(4);            //直接删除元素
    st.erase(st.find(8));           //用find函数找到迭代器，间接删除
    for(set<int>::iterator it=st.begin();it!=st.end();it++)
        printf("%2d",*it);
    st.clear();             //清空
    printf("\n");
//string
    string movie="The Dark Knight";
    string res="nice";
    movie.insert(3,res);        //在movie[3]的位置插入res
printf("%s\n",movie.c_str()); 
							//printf不能直接打印string类的对象，要转换
    movie.erase(movie.begin()+13);      //删除movie[13]位置的元素
    movie.erase(3,4);              //删除从movie[3]开始的长度为4的一段字符
    printf("%s\n",movie.c_str());
    printf("%s\n",movie.substr(4,6).c_str());//从下标4开始的长度为6的子串
    if(movie.find("rk")!=string::npos)
        printf("%2d\n",movie.find("rk"));       //简单的模式匹配
    movie.replace(9,5,"princess"); //替换movie[9]开始的长度为5的子串
    printf("%s\n",movie.c_str());
    movie.clear();
//map
    map<int,char> mp;      						 //键:int,值:char,
    mp[4]='a';mp[2]='b';mp[7]='c';mp[5]='d';
    mp.erase(7);           					 //删除键为7的一个映射
    for(map<int,char>::iterator mi=mp.begin();mi!=mp.end();mi++)
                           					 //迭代器访问
        printf("%d %c\n",mi->first,mi->second);
                       						 //first表示键，second表示值
    map<int,char>::iterator qs=mp.find(5); //用一个迭代器指向键为5的映射
    printf("%d %c\n",qs->first,qs->second);
    printf("%2d\n",mp.size());     			 //映射的对数
    mp.clear();
//queue
    queue<int> Q;              	 //普通队列
    for(int i=1;i<=5;i++)
        Q.push(i);             		 //依次入队
    printf("%2d%2d\n",Q.front(),Q.back()); 			 //打印队头和队尾
    for(int i=1;i<=3;i++)
        Q.pop();               		 //依次出队
    printf("%2d\n",Q.front());
    printf("%2d\n",Q.size());       		//队中元素个数
    if(Q.empty()==true)
        printf("Empty\n");
    else
        printf("Not Empty\n");
    priority_queue<int> q1;        		 //默认为最大优先队列
      //定义一个最小优先队列，为了防止编译器把>>看做移位运算符，中间要有空格
    priority_queue<int,vector<int>,greater<int> > q2;
    q1.push(1);q1.push(5);q1.push(3);q1.push(4);
    q2.push(1);q2.push(5);q2.push(3);q2.push(4);
    printf("%2d%2d\n",q1.top(),q2.top());  		 //top取优先级最高的元素
    for(int i=0;i<2;i++)
    {
        q1.pop();                        //队首(堆顶)元素出队
        q2.pop();
    }
    printf("%2d%2d\n",q1.top(),q2.top());
    f1.name="apple";f1.price=19;f2.name="banana";f2.price=15;f3.name="orange";f3.price=17;
    priority_queue<fruit> q;
    q.push(f1);q.push(f2);q.push(f3);
    printf("%s\n",q.top().name.c_str());    		//打印堆顶水果的名字
//stack
    stack<int> yes;
    for(int i=3;i<=9;i++)
        yes.push(i);            //压入
    for(int i=1;i<=4;i++)
        yes.pop();              //弹出
    printf("%2d%2d%2d\n",yes.top(),yes.size(),yes.empty());
                    				//栈顶元素，栈中元素个数，栈是否为空
//utility
    pair<string,int> p1,p2,p3;      //pair用来构造一个单独的映射
    p1.first="haha";p1.second=5;    //3种构造方法
    p2=make_pair("xixi",55);
    p3=pair<string,int>("heihei",555);
    map<string,int> good;
    good.insert(p1);good.insert(p2);good.insert(p3); //插入到map中
    for(map<string,int>::iterator mi=good.begin();mi!=good.end();mi++)
        printf("%s %d\n",mi->first.c_str(),mi->second);
//algorithm
    int x=1,y=6,z=3;
    printf("%2d%2d%2d\n",max(x,y),min(x,y),max(x,max(y,z)));
    swap(x,z);             							 //交换值
    printf("%2d%2d\n",x,z);
    int a[6]={1,2,3,4,5,6};
    reverse(a,a+4);         					//把a[0]到a[3]反转
    for(int i=0;i<6;i++)
        printf("%2d",a[i]);
    printf("\n");
    fill(a,a+3,0);        					  //把a[0]到a[2]赋值为0
    for(int i=0;i<6;i++)
        printf("%2d",a[i]);
    printf("\n");
    string alpha="abcdefghijk";
    reverse(alpha.begin()+2,alpha.begin()+6);   
                               				 //把alpha[2]到alpha[5]反转
    printf("%s\n",alpha.c_str());
    int b[10]={3,5,2,7,4,1,8,6,9,0};
    sort(b,b+6);              			  //默认为把b[0]到b[5]从小到大排序
    for(int i=0;i<10;i++)
        printf("%2d",b[i]);
    printf("\n");
    string bad[3]={"aaa","bbbb","cc"};
    sort(bad,bad+3,cmp);          				  //按字符串长度排序
    for(int i=0;i<3;i++)
        printf("%s\n",bad[i].c_str());
    return 0;
}
