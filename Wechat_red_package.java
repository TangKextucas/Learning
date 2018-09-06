package practice;

import java.util.Random;
class LeftMoneyPackage {
	public int remainSize;
	public int remainMoney;
	LeftMoneyPackage(int a,int b){
		this.remainSize=a;
		this.remainMoney=b;
	}
}
public class god{
	public static double getRandomMoney(LeftMoneyPackage left){
		if(left.remainSize==1){
			left.remainSize--;
			return (double)Math.round(left.remainMoney*100)/100;
		}
		Random r=new Random();
		double min=0.01;
		double max=left.remainMoney/left.remainSize*2;
		double money=r.nextDouble()*max;
		money=money<=min?0.01:money;
		money=Math.floor(money*100)/100;
		left.remainSize--;
		left.remainMoney-=money;
		return money;
	}
	public static void main(String[] args){
		
		int RedPackage=500,Totalpeople=30;
		LeftMoneyPackage people=new LeftMoneyPackage(Totalpeople,RedPackage);
		for(int i=0;i<Totalpeople;i++){
			System.out.println(getRandomMoney(people));
		}
		
		/*Random f=new Random();
		System.out.println(Math.floor(2.1));
		for(int x=0;x<20;x++)
			System.out.println(f.nextDouble()*(3.5));*/
	}
}