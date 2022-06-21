import numpy as np  # Библиотека для оптимизации вычислений
import pylab as plt  # Библиотека для прорисовки графиков

#==============================================================================
# Загрузка частиц - Определение позиции
#==============================================================================
    
def loadx(bc_particle):            
    global dx, grid_length, rho0, npart, q_over_me, a0
    global charge,mass,wall_left,wall_right
    print ("Load particles")
    if (bc_particle == 1):  
        plasma_start = 0.
        plasma_end = grid_length
        wall_left = 0.
        wall_right = grid_length
    else:
      print("incorrect flag for bc_particle variable")
      
    xload = plasma_end - plasma_start  # расстояние для загрузки
    dpx = xload/npart                    #  среднее расстояние между частицами
    charge = -rho0*dpx         # нормированный заряд псевдочастицы
    mass = charge/q_over_me  #  нормированная масса псевдочастицы (необходимо для анализа кинетической энергии) 

    for i in range(npart):
        x[i] = plasma_start + dpx*(i+0.5)
        x[i] += a0*np.cos(x[i])
        
    return True

    
    
#==============================================================================
#  Запуск частиц и определение скоростей для них 
#==============================================================================
    
def loadv(idist,vte):       
    global npart,v,grid_length,v0
    if idist == 0:  
         v[1:npart] = 0.
    else:
      print("incorrect flag for distribution variable, our programm work only with cold plasma, please set up 0")

# добавление возмущения частицам
    v += v0*np.sin(2*np.pi*x/grid_length)
    return True
    
#==============================================================================
#   Compute densities
#==============================================================================

def density(bc_field,qe):      
   global x,rhoe,rhoi,dx,npart,ngrid,wall_left,wall_right
   j1=np.dtype(np.int32) 
   j2=np.dtype(np.int32) 

   re = qe/dx #  весовой коэффициент заряда 
   rhoe=np.zeros(ngrid+1)        # плотность электронов
    # сопостовление зарядов с заданной сеткой
   for i in range(npart):
      xa = x[i]/dx
      j1 = int(xa)
      j2 = j1 + 1
      f2 = xa - j1
      f1 = 1.0 - f2
      rhoe[j1] = rhoe[j1] + re*f1
      rhoe[j2] = rhoe[j2] + re*f2

   if (bc_field == 1):
      rhoe[0] += rhoe[ngrid]
      rhoe[ngrid] = rhoe[0]
   else:
      print ("Invalid value for bc_field:", bc_field)
      
#  добавление нейтральной плотности ионов   
   rhoi = rho0
   return True

#==============================================================================
#  Вычисление электростатического поля
#==============================================================================

def field():          
  global rhoe,rhoi,ex,dx,ngrid,phi
  
  rhot=rhoe+rhoi  # сетка плотности заряда на заданном промежутке
  
  # интегрирование div.E=rho  (траппециальное приближение) 
  # конечная точка ex=0  (== правая стенка )

  Ex[ngrid]=0.          # нулевое электростатическое поле
  edc = 0.0
  print(rhoi)

  for j in range(ngrid-1,-1,-1):
      Ex[j] = Ex[j+1] - 0.5*( rhot[j] + rhot[j+1] )*dx
      edc = edc + Ex[j]

  if (bc_field == 1):
      Ex[0:ngrid] -= edc/ngrid  
      Ex[ngrid] = Ex[0]
      
  return True
    
    
#==============================================================================
#  Запуск частиц
#==============================================================================

def push():  
    global x,v,Ex,dt,dx,npart,q_over_me

    for i in range(npart):
        #интерполяция графика по известным значениям
      xa = x[i]/dx
      j1 = int(xa)
      j2 = j1 + 1
      b2 = xa - j1
      b1 = 1.0 - b2
      exi = b1*Ex[j1] + b2*Ex[j2]
      v[i] = v[i] + q_over_me*dt*exi       # обновление переменной скорости

    
    x += dt*v    #  обновление частицы (2 часть перемещения) 
    
    return True
    
    
#==============================================================================
#  Проверка граничных условий частиц
#==============================================================================
    
def particle_bc(bc_particle,xl):      
   global x
   if (bc_particle == 1):
     for i in range(npart):
       if ( x[i] < 0.0 ):
         x[i] += xl
       elif ( x[i] >= xl ):
         x[i] -= xl
   else:
     print ("Invalid value for bc_particle:", bc_particle)

   return True
    
    
#==============================================================================
#  Анализ полей и частиц
#==============================================================================
    
def diagnostics():      
    global rhoe,Ex,ngrid,itime,grid_length,rho0,a0
    global ukin, upot, utot, udrift, utherm, emax,fv,fm
    global iout,igraph,iphase,ivdist
    xgrid=dx*np.arange(ngrid+1)
    if (itime==0): 
        plt.figure('fields')
        plt.clf()
    if (igraph > 0):
      if (np.fmod(itime,igraph)==0): # plots every igraph steps
    # Плотность J[A/m^2]
        plt.subplot(2, 2, 1)
        if (itime >0 ): plt.cla()
        plt.plot(xgrid, -(rhoe+rho0), 'r', label='J[A/m^2](x)')
        plt.xlabel('x')
        plt.xlim(0,grid_length)
        plt.ylim(-2*a0,2*a0)
        plt.legend(loc=1)
    # Электрическое поле
        plt.subplot(2, 2, 2)
        if (itime >0 ): plt.cla()
        plt.plot(xgrid, Ex, 'b', label='Ex field')
        plt.xlabel('x')
        plt.ylim(-2*a0,2*a0)
        plt.xlim(0,grid_length)

        plt.legend(loc=1)
    # Phi
        plt.subplot(2,2,4)
        if (itime>0):plt.cla()
        plt.plot(xgrid, phi, 'r', label = 'rhoe')
        plt.xlabel('x')
        plt.ylim(-100, 400)
        plt.xlim(0,grid_length)

        plt.legend(loc=1)
        if (ivdist > 0):
          if (np.fmod(itime,ivdist)==0):
    # график функции распределения для каждого шага
            fv=np.zeros(nvbin+1)        # создание и заполнение нулями массива, в котором будем храниться функция распределения для каждого шага
            dv = 2*vmax/nvbin   # разделение ячеек
            for i in range(npart):
                vax= ( v[i] + vmax )/dv   # скорость сонаправленная с вектором распространения (скорость нормали)
                iv = int(vax)+1 # индекс ячейки 
                if (iv <= nvbin and iv > 0): fv[iv] +=1 #увелечение дистанции, если функция распрделения не ограничена
        
            plt.subplot(2, 2, 4)
            if (itime >0 ): plt.cla()
            vgrid=dv*np.arange(nvbin+1)-vmax
            plt.plot(vgrid, fv, 'g', label='f(v)')
            plt.xlabel('v')
            plt.xlim(-vmax,vmax)
            plt.legend(loc=1)
            fn_vdist = 'vdist_%0*d'%(5, itime)

            np.savetxt(fn_vdist, np.column_stack((vgrid,fv)),fmt=('%1.4e','%1.4e'))   # запись в файл


    
        plt.pause(0.0001)
        plt.draw()
        filename = 'fields_%0*d'%(5, itime)
        if (iout > 0):
          if (np.fmod(itime,iout)==0):  # печать графиков в соответствии с частотй сохранения изображений
            plt.savefig(filename+'.png')

#   Вычисление кинетической энергии
    v2=v**2
    vdrift=sum(v)/npart
    ukin[itime] = 0.5*mass*sum(v2)
    udrift[itime] = 0.5*mass*vdrift*vdrift*npart  
    utherm[itime] = ukin[itime] - udrift[itime]
  
#   Вычисление потенциальной энергии 
    e2=Ex**2
    upot[itime] = 0.5*dx*sum(e2)
    emax = max(Ex) # максимальное поле для нестабильности */
 
#  Суммарная энергия = Потенциальная + Кинетическая 
    utot[itime] = upot[itime] + ukin[itime]
    
    return True
  
#==============================================================================
#    Plot time-histories
#==============================================================================
  
def histories():
    global ukin, upot, utot, udrift, utherm
    xgrid = dt * np.arange(nsteps + 1)
    plt.figure('Energies')
    plt.plot(xgrid, upot, 'red', label='Epot')
    plt.plot(xgrid, ukin, 'green', label='Ekin')
    plt.plot(xgrid, utot, 'black', label='sum')
    plt.xlabel('t')
    plt.ylabel('Energy')
    plt.legend(loc=1)
    plt.savefig('energies.png')

    #   запись в файл energies.out
    np.savetxt('energies.out', np.column_stack((xgrid, upot, ukin, utot)),
               fmt=('%1.4e', '%1.4e', '%1.4e', '%1.4e'))



#==============================================================================
#  Main program
#==============================================================================
  

npart=2048           # количество частиц
ngrid=256             # деление межэлектронного расстояния
nsteps=150           # количество временных шагов на которых проводится вычисление

# particle arrays
x = np.zeros(npart)  # позиция по x 
v = np.zeros(npart) # скорость частиц  		 

# grid arrays
rhoe=np.zeros(ngrid+1)        # плотность электронов 
rhoi=np.zeros(ngrid+1)        # плотность ионов
Ex=np.zeros(ngrid+1)          # электрическое поле 
phi=np.zeros(ngrid + 1)         # потенциал
# time histories
ukin=np.zeros(nsteps+1) # сохранение кинетической энергии за шаг dt
upot=np.zeros(nsteps+1) # сохранение потенциальное энергии за шаг dt
utherm=np.zeros(nsteps+1) # сохранение тепловой скорость в момент dt
udrift=np.zeros(nsteps+1) # сохранение скорости дрифта в момент dt
utot=np.zeros(nsteps+1) # сохранение суммы потенцильной и кинетической энергии за шаг dt
 
grid_length = 4 * np.pi  # межэлектронное расстояние
plasma_start = 0.           # левая границы плазмы
plasma_end = grid_length    # правая граница плазмы
dx = grid_length/ngrid
dt = 0.1            # dt - изменение времени
q_over_me=-1.0       # отношение заряда электрона к массе
rho0 = 1.0           # плотность ионов
vte = 1           # тепловая скорость
nvbin=50            # размер массива графика f(v)
a0 = 0.5            # амплитуда возмущения
vmax = 0.2       # максимальная скорость для графика f(v)
v0=0.0              # скорость возмущения
e00 = 0.0000000000089
wall_left=0.
wall_right=1.
bc_field = 1       #  поле граничного столкновения
bc_particle = 1     # флаг для граничного столкновения частиц
profile = 1
distribution = 0  # флаг для низконемпературной плазмы
ihist = 5         # frequency of time-history output частота вывода времени
igraph = int(np.pi/dt/16)       # частота скриншотов
iphase = igraph
ivdist = -igraph
iout = igraph*1        # частота сохранений изображений
itime = 0         # счетчик времени 

   
   
#  Setup initial particle distribution and fields
#  ----------------------------------------------
   
loadx(bc_particle)              # загрузка частиц в программу 
loadv(distribution,vte)			# define velocity distribution \ определения скорости возмущения
x += 0.5*dt*v                   #  centre positions for 1st leap-frog steps \ центральная для первого шага
particle_bc(bc_particle,grid_length)
density(bc_field, charge)	  	 # compute initial density from particles \ вычисление начальной плотности для частиц
field()			            # compute initial electric field \ вычисление электрического поля
diagnostics()		            # output initial conditions \ вычисление начальных условий
print ('resolution dx/\lambda_D=',dx/vte) \

#  Main iteration loop
#  -------------------

for itime in range(1,nsteps+1):
    print ('timestep ',itime) # вывод времени для отладки программы
    push()			       # загрузка частиц 
    particle_bc(bc_particle,grid_length) # применение граничных условий для частиц
    density(bc_field,charge)	    # вычисление плотности
    field()		               # вычисление электрического поля из уравнения Пуассона
    diagnostics()	# скриншот и сохранение данных в массив

histories()   # сохранение всех полученных графиков
      
#raw_input("Press key to finish")
#plt.close()
print ('done')