/*这是一个新的NACA类，用新的算法生成翼型点的数据 */
class generateNaca extends Body{
    int n = 200;//机翼上下表面的点的数量
    float []x = new float[n];
    float []y = new float[n];
    float []yt = new float[n];//机翼不同位置的厚度
    float []xu = new float[n];
    float []yu = new float[n];
    float []xl = new float[n];
    float []yl = new float[n];
    float []theta = new float[n];
    float c;
    generateNaca(float xc,float yc,float name1,float name2,float name3,float c,Window window){
        super(xc,yc,window);
        float M =(name1)/100;
        float P = 10*name2;
        P = P/100;
        float T = name3;
        T = T/100;
        for(int i=0;i<n;i++){
            x[i] = i*1.0/(n-1);
            if(x[i] <= P){
                y[i] = M/(pow(P,2))*(2*P*x[i] - pow(x[i],2));
            }
            else{
                y[i] = M / (pow(1-P,2))*((1-2*P)+2*P*x[i]-pow(x[i],2));
            }
            yt[i] = T/0.2*(0.2969*sqrt(x[i])-0.1260*x[i]-0.3516*pow(x[i],2)+0.2843*pow(x[i],3)-0.1015*pow(x[i],4));
            

        }
        for(int i=0;i<n;i++){
            if(i==n-1){
                theta[i] = theta[i-1];
            }
            else{
                theta[i] = atan(((y[i + 1] - y[i])/(1.0/(n-1))));
            }
             xu[i] = x[i] - yt[i] * sin(theta[i]);
             yu[i] = y[i] + yt[i] * cos(theta[i]);
             xl[i] = x[i] + yt[i] * sin(theta[i]);
             yl[i] = y[i] - yt[i] * cos(theta[i]);
        }
        for(int i=0;i<n;i++){
            add(xc+xu[i]*c,yc+yu[i]*c);
        }
        for(int i=0;i<n-2;i++){
            add(xc+xl[n-i-1]*c,yc+yl[n-i-1]*c);
        }
        end();
        this.c = c;
    }
}
