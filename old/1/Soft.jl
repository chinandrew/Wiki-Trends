
c= 0
for i in (t_0+1):t
	c+ u[0:t_0]' * b+rho*b'*(L*a)[0:t_0]
end
c = c/(t-t_0)*2/rho



function soft(a,u,rho, lambda,t,t_0)
    c= zeros(t_0)
    for i in (t_0+1):(t-1)
        c+ u[1:t_0]+rho*(L*a)[1:t_0]
    end
    c = c/(t-t_0)*2/rho
    return sign(c).*max(abs(c)-lambda/2,0)
end

#

#ifelse(c.<lambda/2, c+0.5*lambda, ifelse(c.>lambda/2, c-0.5*lambda, 0))