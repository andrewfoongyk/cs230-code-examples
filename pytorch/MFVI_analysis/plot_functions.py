import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm

def g(r):
    return 1/(r * np.sqrt(2 * np.pi)) * np.exp(-r**2/2) + norm.cdf(r) 

def f(r):
    return (1 + 1/(r**2)) * norm.cdf(r) +  1/(r * np.sqrt(2 * np.pi)) * np.exp(-r**2/2) - g(r)**2

def E_phi(mu, sigma):
    r = mu/sigma
    return mu * g(r)

def Var_phi(mu, sigma):
    M = mu*norm.cdf(mu/sigma) + sigma*norm.pdf(mu/sigma)
    #r = mu/sigma 
    #return mu**2 * f(r) 
    return (mu**2 + sigma**2)*norm.cdf(mu/sigma) + mu*sigma*norm.pdf(mu/sigma) - M**2

def linearised_Var_phi(mu, sigma):
    return np.heaviside(mu, 0) * sigma**2

def compare_var_linearised(muU, muv, varU, varv, title='', show=False, xrange=[-4, 4], check_pass_test=False, plot=False):
    # plot true var[phi] versus linearised approx
    plt.figure() 
    x = np.linspace(xrange[0], xrange[1], 500)
    mu = muU*x + muv 
    sigma = np.sqrt(varU*x**2 + varv)
    # true variance
    Varphi = Var_phi(mu, sigma)
    varphiline, = plt.plot(x, Varphi, label='Var $[\phi(a)]$')
    # linearised variance
    linVarphi = linearised_Var_phi(mu, sigma)
    linvarphiline, =  plt.plot(x, linVarphi, label='truncated quadratic approximation', ls='--')

    if plot == True:
        # plot vertical line showing where the ReLU kink is, for the mean value of the weights
        #kink = -muv/muU 
        #plt.axvline(x=kink, ls='--', color='k')
        plt.xlabel('$x$')
        plt.xlim(xrange[0], xrange[1])
        plt.ylabel('variance')
        plt.grid(True)
        plt.legend(handles = [varphiline, linvarphiline])
        plt.title('True and approximate variance when $\mu_U =${}, $\mu_v =${}, $\sigma_U^2 =${}, $\sigma_v^2 =${}'.format(muU, muv, varU, varv))
        if show:
            plt.show()
        else:
            plt.savefig('plots//wing_condition//plot_true_vs_approx_var_' + title + '.pdf')
            plt.close()

    if check_pass_test == True:
        # check if the conditions for David's proof hold
        # import pdb; pdb.set_trace()

        x_S1 = np.linspace(0, 10, 1000)
        mu_S1 = muU*x_S1 + muv 
        sigma_S1 = np.sqrt(varU*x_S1**2 + varv)
        x_S2 = np.linspace(-10, 0, 1000)
        mu_S2 = muU*x_S2 + muv 
        sigma_S2 = np.sqrt(varU*x_S2**2 + varv)
        
        Varphi_S1 = Var_phi(mu_S1, sigma_S1)
        Varphi_S2 = Var_phi(mu_S2, sigma_S2)

        x_S2_wing = -x_S1
        mu_S2_wing = muU*x_S2_wing + muv 
        sigma_S2_wing = np.sqrt(varU*x_S2_wing**2 + varv)
        Varphi_S2_wing = Var_phi(mu_S2_wing, sigma_S2_wing)

        Varphi_S1 = np.maximum(Varphi_S1, 0)
        Varphi_S2 = np.maximum(Varphi_S2, 0)
        Varphi_S2_wing = np.maximum(Varphi_S2_wing, 0)

        # check if in S1          
        x_0 = Varphi_S1[0]
        x_greater = Varphi_S1[1:]
        x_greater_min = np.maximum(0., np.amin(x_greater))
        if x_0 > x_greater_min:
            in_S1 = False
            #print('Not in S1')
        else:
            in_S1 = True
            #print('In S1')
            # check if the wings are at least as large  
            violated_S1 = np.any(np.greater(Varphi_S2_wing, Varphi_S1))
            if violated_S1 == True:
                pass
                #import pdb; pdb.set_trace()
                #print('violated wing condition')
            else:
                pass
                #print('wing condition satisfied')

        # check if in S2          
        x_0 = Varphi_S2[-1]
        x_less = Varphi_S2[:-1]
        x_less_min = np.maximum(0., np.amin(x_less))
        if x_0 > x_less_min:
            in_S2 = False
            #print('Not in S2')
        else:
            in_S2 = True
            #print('In S2')
            violated_S2 = np.any(np.greater(Varphi_S1, Varphi_S2_wing))
            if violated_S2 == True:
                pass
                #import pdb; pdb.set_trace()
                #print('violated wing condition')           
            else:
                pass
                #print('wing condition satisfied')

        if (in_S1 == True) and (in_S2 == False):
            print('check')
            if violated_S1 == True:
                print('violated wing condition for S1')
                import pdb; pdb.set_trace()

        if (in_S2 == True) and (in_S1 == False):
            print('check')
            if violated_S2 == True:
                print('violated wing condition for S2')
                import pdb; pdb.set_trace()

        if (in_S2 == True) and (in_S1 == True):
            print('check')
            if (violated_S1 == True) and (violated_S2 == True):
                print('violated wing condition for S1 and S2')
                import pdb; pdb.set_trace()
            if (violated_S1 == False) and (violated_S2 == False):
                print ('satisfies both wing conditions simultaneously!')
                # import pdb; pdb.set_trace()

        if (in_S1 == True) and (in_S2 == True):
            in_both = True
        else:
            in_both = False

        if (in_S1 == False) and (in_S2 == False): 
            import pdb; pdb.set_trace()
            return False, in_both 
        else:
            return True, in_both

if __name__ == '__main__':
    r = np.linspace(-5, 5, 1000)
    g_out = g(r) 
    f_out = f(r) 
    
    # plt.figure(1)
    # plt.plot(r, g_out) 
    # plt.xlabel('r')
    # plt.ylabel('g(r)')
    # plt.ylim(-5, 5)
    # plt.grid(True)
    # plt.savefig('plots//plot_g.pdf')
    # plt.close()

    # plt.figure(2)
    # plt.plot(r, f_out) 
    # plt.xlabel('r')
    # plt.ylabel('f(r)')
    # plt.ylim(0, 5)
    # plt.grid(True)
    # plt.savefig('plots//plot_f.pdf')
    # plt.close()

    # plt.figure(3)
    # plt.scatter(1/r, f_out) 
    # plt.xlabel('r')
    # plt.ylabel('f(1/r)')
    # #plt.ylim(0, 5)
    # plt.grid(True)
    # plt.savefig('plots//plot_f_one_over_r.pdf')
    # plt.close()

    # mu = np.linspace(-5, 5, 1000)
    # sigma = 1 #################### this is unrealistic as mu and sigma should be coupled
    # plt.figure(4)
    # Ephi = E_phi(mu, sigma) 
    # plt.plot(mu, Ephi) 
    # plt.xlabel('$\mu$')
    # plt.ylabel('Expected value of phi for sigma = 1')
    # #plt.ylim(0, 5)
    # plt.grid(True)
    # plt.savefig('plots//plot_Ephi.pdf')
    # plt.close()

    # plt.figure(5)
    # Varphi = Var_phi(mu, 2) 
    # plt.plot(mu, Varphi) 
    # plt.xlabel('$\mu$')
    # plt.ylabel('Variance of phi for sigma = 1')
    # #plt.ylim(0, 5)
    # plt.grid(True)
    # plt.savefig('plots//plot_Varphi.pdf')
    # plt.close()

    # plt.figure(6)
    # x = np.linspace(-2, 2, 500)
    # muU = 3
    # muv = 3
    # varU = 0.1
    # varv = 0.01
    # mu = muU*x + muv 
    # sigma = np.sqrt(varU*x**2 + varv)
    # Varphi = Var_phi(mu, sigma)
    # plt.plot(x, Varphi) 
    # plt.xlabel('$x$')
    # plt.ylabel('Variance of phi for some weights')
    # plt.grid(True)
    # plt.savefig('plots//plot_Varphi_of_x.pdf')
    # plt.close()

    # plt.figure(7) # f over g as a function of x
    # r = mu/sigma
    # f_out = f(r)
    # g_out = g(r) 
    # plt.scatter(x, g_out/np.sqrt(f_out)) 
    # plt.xlabel('$x$')
    # plt.ylabel('$\sqrt{f}/g$ for some weights')
    # plt.grid(True)
    # plt.savefig('plots//plot_f_over_g_of_x.pdf')
    # plt.close()

    # plt.figure(8) # true var[phi] versus linearised approx
    # x = np.linspace(-2, 2, 500)
    # muU = 3
    # muv = 3
    # varU = 0.1
    # varv = 0.01
    # mu = muU*x + muv 
    # sigma = np.sqrt(varU*x**2 + varv)
    # # true variance
    # Varphi = Var_phi(mu, sigma)
    # varphiline, = plt.plot(x, Varphi, label='Var $[\phi(a)]$')
    # # linearised variance
    # linVarphi = linearised_Var_phi(mu, sigma)
    # linvarphiline, =  plt.plot(x, linVarphi, label='truncated quadratic approximation')
    # plt.xlabel('$x$')
    # plt.ylabel('variance')
    # plt.grid(True)
    # plt.legend(handles = [varphiline, linvarphiline])
    # plt.title('True versus approximate variance when $\mu_U =${}, $\mu_v =${}, $\sigma_U^2 =${}, $\sigma_v^2 =${}'.format(muU, muv, varU, varv))
    # plt.savefig('plots//plot_true_vs_approx_var.pdf')
    # plt.close()

    # compare many linearised and exact non-convex variance contributions
    final_pass_test = True
    count = 0
    for muU in [0]: #[-10, -3, -2, -1, -0.01]:
        for muv in [-10, -3, -2, -1, -0.01]:
            for varU in [1e-10, 0.01, 0.1, 1, 10]:
                for varv in [1e-10, 0.01, 0.1, 1, 10]:
                    pass_test, in_both = compare_var_linearised(muU=muU, muv=muv, varU=varU, varv=varv, title=str(count), check_pass_test=True, plot=True)
                    count = count + 1
                    if in_both == True:
                        pass
                        #print('muU:{}, muv:{}, varU:{}, varv:{}'.format(muU, muv, varU, varv))
                    if pass_test == False:
                        final_pass_test = False
                        # import pdb; pdb.set_trace()
    print('Final check:{}'.format(final_pass_test))
    
    # muU = 3
    # muv = -3
    # varU = 10
    # varv = 0.01
    # compare_var_linearised(muU=muU, muv=muv, varU=varU, varv=varv, title='sandbox', xrange=[-2, 2])

    # test a few
    # muU = -1
    # muv = 3
    # varU = 1
    # varv = 0.01
    # compare_var_linearised(muU=muU, muv=muv, varU=varU, varv=varv, show=True, xrange=[-200, 200])