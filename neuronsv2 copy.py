from matplotlib import pylab as plt
import numpy as np
import time

def neuron():
    #Time
    Time = 10000 #ms
    dt=.01
    nsteps = int(Time/dt)
    t=np.linspace(0,Time,nsteps)


    #parameters#
    gbar_K=36
    gbar_Na=120
    g_L=.3
    E_K = -12
    E_Na=115
    E_L=10.6
    C=1

    #External currents#
    mu = 6.5
    sigma = 2
    I=np.random.normal(mu, sigma, size = nsteps)

    mu2 = 6.5
    sigma2 = 2
    I2=np.random.normal(mu2, sigma2, size = nsteps)


    #Inhibition#
    a1 = 15
    b1 = 20/dt


    a2 = 15
    b2 = 20/dt

    #Resting potentials
    r1 = -50  #  -65
    r2 = 0  #-70




    #_________NEURON 1__________#



    #set up arrays#

    V=np.zeros((nsteps))
    m = np.zeros((nsteps))
    n =  np.zeros((nsteps))
    h =  np.zeros((nsteps))
    in1 = np.zeros((nsteps))
    in1[0] = 0
    V[0]=r1
    alpha_n = np.zeros((nsteps))
    beta_n = np.zeros((nsteps))
    alpha_m = np.zeros((nsteps))
    beta_m = np.zeros((nsteps))
    alpha_h = np.zeros((nsteps))
    beta_h = np.zeros((nsteps))
    alpha_n[0] = .01 * ( (10-V[0]) / (np.exp((10-V[0])/10)-1) )
    beta_n[0] = .125*np.exp(-V[0]/80)
    alpha_m[0] = .1*( (25-V[0]) / (np.exp((25-V[0])/10)-1) )
    beta_m[0] = 4*np.exp(-V[0]/18)
    alpha_h[0] = .07*np.exp(-V[0]/20)
    beta_h[0] = 1/(np.exp((30-V[0]/10)+1))

    n[0] = alpha_n[0]/(alpha_n[0]+beta_n[0])
    m[0] = alpha_m[0]/(alpha_m[0]+beta_m[0])
    h[0] = alpha_h[0]/(alpha_h[0]+beta_h[0])
    fires1 = 0
    t1initial = 0

    numbouts1 =np.zeros((nsteps))
    nb1 = 0
    boutlengths1 = np.zeros((nsteps))
    bnum1 = 1
    boutcounter1 = 0
    pr1 = np.zeros(int(nsteps))




    #_________NEURON 2___________#




    #initial#

    V2 = np.zeros((nsteps))
    m2  = np.zeros((nsteps))
    n2 =  np.zeros((nsteps))
    h2 =  np.zeros((nsteps))
    in2 = np.zeros((nsteps))
    V2[0]=r2
    alpha_n2 = np.zeros((nsteps))
    beta_n2 = np.zeros((nsteps))
    alpha_m2 = np.zeros((nsteps))
    beta_m2 = np.zeros((nsteps))
    alpha_h2 = np.zeros((nsteps))
    beta_h2 = np.zeros((nsteps))
    alpha_n2[0] = .01 * ( (10-V2[0]) / (np.exp((10-V2[0])/10)-1) )
    beta_n2[0] = .125*np.exp(-V2[0]/80)
    alpha_m2[0] = .1*( (25-V2[0]) / (np.exp((25-V2[0])/10)-1) )
    beta_m2[0] = 4*np.exp(-V2[0]/18)
    alpha_h2[0] = .07*np.exp(-V2[0]/20)
    beta_h2[0] = 1/(np.exp((30-V2[0])/10)+1)
    n2[0] = alpha_n2[0]/(alpha_n2[0]+beta_n2[0])
    m2[0] = alpha_m2[0]/(alpha_m2[0]+beta_m2[0])
    h2[0] = alpha_h2[0]/(alpha_h2[0]+beta_h2[0])
    fires2 = 0
    t2initial = 0

    numbouts2 = np.zeros((nsteps))
    nb2 = 0
    boutlengths2 = np.zeros((nsteps))
    bnum2 = 1
    boutcounter2 = 0
    pr2 = np.zeros((int(nsteps)))


 


    #Euler's#


    #_________NEURON1_____________#


    for i in range(nsteps-1):



     #Check for Inhibition#
        if fires2 == 1:
            fires2 = 0
            if i+b1 <=  len(t) - 1:
                for j in range(int(i),int(i+b1)):
                    in1[j] = in1[j] - a1

            else:
                for j in range(i,len(t) - 10):
                    in1[j] = in1[j] - a1

        #coefficients at each t
        V[i] = V[i]
        alpha_n[i] = .01 * ( (10-V[i]) / (np.exp((10-V[i])/10)-1) )
        beta_n[i] = .125*np.exp(-V[i]/80)
        alpha_m[i] = .1*( (25-V[i]) / (np.exp((25-V[i])/10)-1) )
        beta_m[i] = 4*np.exp(-V[i]/18)
        alpha_h[i] = .07*np.exp(-V[i]/20)
        beta_h[i] = 1/(np.exp((30-V[i])/10)+1)


        #current calculation#
        I_Na = (m[i]**3) * gbar_Na * h[i] * (V[i]-E_Na)
        I_K = (n[i]**4) * gbar_K * (V[i]-E_K)
        I_L = g_L *(V[i]-E_L)
        I_ion = I[i] - I_K - I_Na - I_L + in1[i] + I_lc[i]


        #Euler's approximation#
        V[i+1] = V[i] + dt*I_ion/C
        n[i+1] = n[i] + dt*(alpha_n[i] *(1-n[i]) - beta_n[i] * n[i])
        m[i+1] = m[i] + dt*(alpha_m[i] *(1-m[i]) - beta_m[i] * m[i])
        h[i+1] = h[i] + dt*(alpha_h[i] *(1-h[i]) - beta_h[i] * h[i])

        if V[i] >= 50 and V[i-1] < 50:
            fires1 = 1
            bout1 = 1
            if t1initial== 0:
                t1initial = i
            if t2initial > 0:
               boutlengths2[bnum2] = i - t2initial
               bnum2 = bnum2 + 1
               t2initial = 0



        #_________NEURON2__________#

        #Check for Inhibition#
        if fires1 == 1:
            fires1 = 0
            if i+b2 <=  len(t) - 1:
                for k in range(int(i),int(i+b2)):
                    in2[k] = in2[k] - a2
            else:
               for k in range(i, len(t) - 1):
                 in2[k] = in2[k] - a2

        #coefficients at each time#
        V2[i] = V2[i]
        alpha_n2[i] = .01 * ( (10-V2[i]) / (np.exp((10-V2[i])/10)-1) )
        beta_n2[i] = .125*np.exp(-V2[i]/80)
        alpha_m2[i] = .1*( (25-V2[i]) / (np.exp((25-V2[i])/10)-1) )
        beta_m2[i] = 4*np.exp(-V2[i]/18)
        alpha_h2[i] = .07*np.exp(-V2[i]/20)
        beta_h2[i] = 1/(np.exp((30-V2[i])/10)+1)


        #currents#
        I_Na2 = (m2[i]**3) * gbar_Na * h2[i] * (V2[i]-E_Na)
        I_K2 = (n2[i]**4) * gbar_K * (V2[i]-E_K)
        I_L2 = g_L *(V2[i]-E_L)
        I_ion2 = I2[i] - I_K2 - I_Na2 - I_L2 + in2[i]


        #Euler's approximation#
        V2[i+1] = V2[i] + dt*I_ion2/C
        n2[i+1] = n2[i] + dt*(alpha_n2[i] *(1-n2[i]) - beta_n2[i] * n2[i])
        m2[i+1] = m2[i] + dt*(alpha_m2[i] *(1-m2[i]) - beta_m2[i] * m2[i])
        h2[i+1] = h2[i] + dt*(alpha_h2[i] *(1-h2[i]) - beta_h2[i] * h2[i])

        if V2[i] >= 50 and V2[i-1] < 50:
            bout1 = 0
            fires2 = 1
            if t2initial == 0:
               t2initial = i
            if t1initial > 0:
               boutlengths1[bnum1] = i - t1initial
               bnum1 = bnum1 + 1
               t1initial = 0
        if i ==  nsteps-2:
            if t1initial > 0:
                boutlengths1[bnum1] = i - t1initial
            if t2initial > 0:
                boutlengths2[bnum2] = i - t2initial 


    boutlengths1 = boutlengths1[boutlengths1 != 0]
    boutlengths2 = boutlengths2[boutlengths2 != 0]

    boutlengths1 = boutlengths1*dt
    boutlengths2 = boutlengths2*dt
    numboutlengths1 = len(boutlengths1)
    numboutlengths2 = len(boutlengths2)



    # #calculating probability of bouts#
    boutlengths1 = boutlengths1/dt
    boutlengths2 = boutlengths2/dt

    for d in range(0,nsteps):
        for k in range(0, numboutlengths1):
            if boutlengths1[k] > d:
                nb1 = nb1 + 1
        numbouts1[d] = nb1
        nb1 = 0
        if numbouts1[d] == 0:
            pr1[d] = 0
        else:
            pr1[d] = numbouts1[d]/numboutlengths1

        for j in range(0,numboutlengths2):
            if boutlengths2[j] > d:
                nb2 = nb2 + 1

        numbouts2[d] = nb2
        nb2 = 0
        if numbouts2[d] == 0:
            pr2[d] = 0
        else:
            pr2[d] = numbouts2[d]/numboutlengths2

    numbouts1 = numbouts1[numbouts1 != 0]
    numbouts2 = numbouts2[numbouts2 != 0]
    pr1 = pr1[pr1 != 0]
    pr2 = pr2[pr2 != 0]

    boutlengths1 = boutlengths1*dt
    boutlengths2 = boutlengths2*dt


    mean1 = np.mean(boutlengths1)
    var1 = np.var(boutlengths1)
    mean2 = np.mean(boutlengths2)
    var2 = np.var(boutlengths2)
    print("mean 1: ", mean1)
    #print("variance 1: ", var1)
    print("mean 2: ", mean2)
    #print("variance 2: ", var2)
    logt = np.log(np.linspace(1,len(pr1),len(pr1)))



    #PLOTS


    #____NEURON1_______#
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.plot(t, V, )
    plt.plot(t, in1)
    plt.legend({'voltage', 'inhibition'})
    plt.ylabel('Voltage (mv)')
    plt.title('Wake Neuron')


    # # #_____NEURON2_______#
    plt.subplot(2,2,2)
    plt.plot(t, V2)
    plt.plot(t, in2)
    plt.legend({'voltage', 'input current'})
    plt.ylabel('Voltage (mv) - Sleep')
    plt.xlabel('time (ms)')
    plt.title('Sleep Neuron')








    #plotting pr(bout)#

    plt.subplot(2,2,3)
    plt.plot(np.log(pr1))
    plt.ylabel('(Log) pr of bout 1')
    plt.xlabel('time (ms)')
    #plt.title('Bout distributions (1)')


    plt.subplot(2,2,4)
    plt.plot(np.log(pr2))
    plt.ylabel('(Log) pr of bout 2')
    plt.xlabel('time (ms)')
    #plt.title('Bout distributions (2)')

    #Log-log plot
    plt.figure(3)
    plt.plot(logt, np.log(pr1))
    plt.ylabel('Voltage (mv) - Wake')
    plt.title('Log-log plot')



    plt.show()

neuron()
