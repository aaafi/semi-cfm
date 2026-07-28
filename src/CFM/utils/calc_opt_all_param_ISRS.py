import numpy as np

def calc_opt_one_point(sigma_0,deltaz,Pout_dBm,N_ch,z,m_pow,alpha_0):


    sigma = sigma_0
    N_z2 = len(z)
    P_0 = np.ones((N_z2,1))*(10**(-3+0.1*Pout_dBm[0,:]))
    P = (10**(-3+0.1*Pout_dBm))
    z = z.reshape((z.size , 1))
    z_f = z@np.ones((1,N_ch))
    sigma_lim = sigma
    sigma_lim = sigma_lim.reshape((1 , sigma_lim.size ))
    sig_2D = np.ones((N_z2,1))@sigma_lim
    h =(np.exp(-sig_2D*z_f)-1)/sig_2D
    #if sigma = 0, (exp(-sig_2D.*z_f)-1)./(-sig_2D) = z_f;
    h[np.where(sigma==0)] = -z_f[np.where(sigma==0)]

    # the terms in the matrix of Eq.(30.2), which is re-writen in file
    # 'CFM=backward-RA'
    a_0 = np.sum(2*z_f**2*P**m_pow,axis=0)            # a
    a_0_prim = np.sum(2*z_f*h*P**m_pow,axis=0)       # -b (-c)
    a_1 = np.sum(-2*z_f*h*P**m_pow,axis=0)            #b (c)
    a_1_prim = np.sum(-2*h**2*P**m_pow,axis=0)        # -d
    b_1 = np.sum(-z_f*P**m_pow*(np.log(P/P_0)),axis=0) # e
    b_2 = np.sum(-h*P**m_pow*(np.log(P/P_0)) , axis=0)  # -f


    # alfa0 = (de-bf)/(ad-bc)
    alfa_0_opt = ((a_1_prim*b_1)-(a_1*b_2))/((a_0*a_1_prim)-(a_1*a_0_prim))
    alfa_0_opt = alfa_0_opt.reshape((1 , N_ch))


    tss=0.75
    for i in range(alpha_0.size):
        if alfa_0_opt[0,i] < (tss*alpha_0[i]):
            alfa_0_opt[0,i] = tss*alfa_0_opt[0,i]


    # Eq. (29)
    alfa_1_opt = -np.sum(-h*P**m_pow*((np.log(P/P_0))+2*z_f*(np.ones((N_z2,1))*alfa_0_opt)),axis=0)/np.sum(2*h**2*P**m_pow,axis=0)

    # calculate the cost function
    sigma_opt = sigma
    alfa_0_2D = np.ones((N_z2,1))*alfa_0_opt
    alfa_1_2D = np.ones((N_z2,1))*alfa_1_opt
    cost_fun_opt = np.sum((P**m_pow*(np.log(P/P_0)+2*alfa_0_2D*z_f+2*(alfa_1_2D/sig_2D)-2*(alfa_1_2D/sig_2D)*np.exp(-sig_2D*z_f))**2),axis=0)

    return alfa_0_opt, alfa_1_opt, sigma_opt, cost_fun_opt


def calc_opt_all_param_ISRS(alpha_0_vec_freq_dep,deltaz,Pout_dBm,N_ch,z,m_pow,opt_loop):
    alfa_0_opt = np.empty((1 , N_ch), dtype="float" ) 
    alfa_1_opt = np.zeros((1 , N_ch)) 
    sigma_opt = np.zeros((1 , N_ch))
    cost_fun_mat = np.zeros((4 , N_ch))
    alfa_0_mat = np.zeros((4 , N_ch))
    alfa_1_mat = np.zeros((4 , N_ch))
    sigma_mat = np.zeros((4 , N_ch))
    sigma_0 = 2*alpha_0_vec_freq_dep
    if opt_loop == 1 :
        alfa_0_opt, alfa_1_opt, sigma_opt, cost_fun_opt = calc_opt_one_point(sigma_0,deltaz,Pout_dBm,N_ch,z,m_pow,alpha_0_vec_freq_dep)
    else:
        # Golden section search
        # https://en.wikipedia.org/wiki/Golden-section_search
        # diff([X_1,X_2,X_3,X_4]) = phi:1:phi    
        phi = (1+np.sqrt(5))/2    # golden ratio
        r = 1/(1+(1/phi))      # r = phi-1
        # lower bound for search
        X_1 = 0.5*sigma_0    
        # the point inbtween X_1 and X_4, but close to X_4
        X_3 = 5*sigma_0
        # upper bound for search
        X_4 = X_1+(X_3-X_1)/r
        # the point inbtween X_1 and X_4, but close to X_2
        X_2 = X_4-r*(X_4-X_1)
        
        # for each point, alfa0,alfa1,sigma,cost_fun are calculated  
        alfa_0_1, alfa_1_1, sigma_1, cost_fun_1 = calc_opt_one_point(X_1,deltaz,Pout_dBm,N_ch,z,m_pow,alpha_0_vec_freq_dep)
        alfa_0_2, alfa_1_2, sigma_2, cost_fun_2 = calc_opt_one_point(X_2,deltaz,Pout_dBm,N_ch,z,m_pow,alpha_0_vec_freq_dep)
        alfa_0_3, alfa_1_3, sigma_3, cost_fun_3 = calc_opt_one_point(X_3,deltaz,Pout_dBm,N_ch,z,m_pow,alpha_0_vec_freq_dep)
        alfa_0_4, alfa_1_4, sigma_4, cost_fun_4 = calc_opt_one_point(X_4,deltaz,Pout_dBm,N_ch,z,m_pow,alpha_0_vec_freq_dep)

        cost_fun_mat[0 , : ]=cost_fun_1
        cost_fun_mat[1 , : ]=cost_fun_2
        cost_fun_mat[2 , : ]=cost_fun_3
        cost_fun_mat[3 , : ]=cost_fun_4

        
        # the target is to find out the minimum cost function and its
        # alfa0,alfa1,sigma
        val_min = np.min(cost_fun_mat, axis=0)
        loc_min = np.argmin(cost_fun_mat, axis=0) 
        #[val_max, loc_max] = max(cost_fun_mat)
        
        for nn in range(opt_loop-1):
            C_1 = (loc_min==0)
            C_2 = (loc_min==1)
            C_3 = (loc_min==2)
            C_4 = (loc_min==3)

            new_X_1=(C_1+C_2)*(X_3-r*(X_3-X_1))+(C_3+C_4)*(X_2+r*(X_4-X_2))
            alfa_0_new_1, alfa_1_new_1, sigma_new_1, cost_fun_new_1 = calc_opt_one_point(new_X_1,deltaz,Pout_dBm,N_ch,z,m_pow,alpha_0_vec_freq_dep)
            
            X_1_N = (C_1+C_2)*X_1+(C_3+C_4)*X_2
            alfa_0_1_N=(C_1+C_2)*alfa_0_1+(C_3+C_4)*alfa_0_2
            alfa_1_1_N=(C_1+C_2)*alfa_1_1+(C_3+C_4)*alfa_1_2
            sigma_1_N=(C_1+C_2)*sigma_1+(C_3+C_4)*sigma_2
            cost_fun_1_N=(C_1+C_2)*cost_fun_1+(C_3+C_4)*cost_fun_2
            
            X_2_N=(C_1+C_2)*(X_3-r*(X_3-X_1))+(C_3+C_4)*X_3
            alfa_0_2_N=(C_1+C_2)*alfa_0_new_1+(C_3+C_4)*alfa_0_3
            alfa_1_2_N=(C_1+C_2)*alfa_1_new_1+(C_3+C_4)*alfa_1_3
            sigma_2_N=(C_1+C_2)*sigma_new_1+(C_3+C_4)*sigma_3
            cost_fun_2_N=(C_1+C_2)*cost_fun_new_1+(C_3+C_4)*cost_fun_3
            
            X_3_N=(C_1+C_2)*X_2+(C_3+C_4)*(X_2+r*(X_4-X_2))
            alfa_0_3_N=(C_1+C_2)*alfa_0_2+(C_3+C_4)*alfa_0_new_1
            alfa_1_3_N=(C_1+C_2)*alfa_1_2+(C_3+C_4)*alfa_1_new_1
            sigma_3_N=(C_1+C_2)*sigma_2+(C_3+C_4)*sigma_new_1
            cost_fun_3_N=(C_1+C_2)*cost_fun_2+(C_3+C_4)*cost_fun_new_1
            
            X_4_N=(C_1+C_2)*X_3+(C_3+C_4)*X_4
            alfa_0_4_N=(C_1+C_2)*alfa_0_3+(C_3+C_4)*alfa_0_4
            alfa_1_4_N=(C_1+C_2)*alfa_1_3+(C_3+C_4)*alfa_1_4
            sigma_4_N=(C_1+C_2)*sigma_3+(C_3+C_4)*sigma_4
            cost_fun_4_N=(C_1+C_2)*cost_fun_3+(C_3+C_4)*cost_fun_4
            
            X_1=X_1_N
            alfa_0_1=alfa_0_1_N
            alfa_1_1=alfa_1_1_N
            sigma_1=sigma_1_N
            cost_fun_1=cost_fun_1_N
            
            X_2=X_2_N
            alfa_0_2=alfa_0_2_N
            alfa_1_2=alfa_1_2_N
            sigma_2=sigma_2_N
            cost_fun_2=cost_fun_2_N
            
            X_3=X_3_N
            alfa_0_3=alfa_0_3_N
            alfa_1_3=alfa_1_3_N
            sigma_3=sigma_3_N
            cost_fun_3=cost_fun_3_N
            
            X_4=X_4_N
            alfa_0_4=alfa_0_4_N
            alfa_1_4=alfa_1_4_N
            sigma_4=sigma_4_N
            cost_fun_4=cost_fun_4_N

            cost_fun_mat[0 , : ]=cost_fun_1
            cost_fun_mat[1 , : ]=cost_fun_2
            cost_fun_mat[2 , : ]=cost_fun_3
            cost_fun_mat[3 , : ]=cost_fun_4
            
            val_min = np.min(cost_fun_mat, axis=0)
            loc_min = np.argmin(cost_fun_mat, axis=0)
            #[val_max loc_max]=max(cost_fun_mat)
            
        # val_min;
        alfa_0_mat[0 , : ]=alfa_0_1
        alfa_0_mat[1 , : ]=alfa_0_2
        alfa_0_mat[2 , : ]=alfa_0_3
        alfa_0_mat[3 , : ]=alfa_0_4

        alfa_1_mat[0 , : ]=alfa_1_1
        alfa_1_mat[1 , : ]=alfa_1_2
        alfa_1_mat[2 , : ]=alfa_1_3
        alfa_1_mat[3 , : ]=alfa_1_4
        
        sigma_mat[0 , : ]=sigma_1
        sigma_mat[1 , : ]=sigma_2
        sigma_mat[2 , : ]=sigma_3
        sigma_mat[3 , : ]=sigma_4
        

        I = loc_min 
        for i in range(N_ch) :
            alfa_0_opt[0 , i ] = alfa_0_mat[I[i] , i ]
            alfa_1_opt[0 , i ] = alfa_1_mat[I[i] , i ]
            sigma_opt[0 , i ]  = sigma_mat[I[i] , i]           


    return alfa_0_opt, alfa_1_opt, sigma_opt