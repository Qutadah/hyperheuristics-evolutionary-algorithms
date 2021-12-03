import time
import random
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
#import functools.reduce

#from inliner import inline


start_time = time.time()

#create services dictionary

#we inserted the locations as time, so we can calculate easier

#Prozess, Ressource, Alternativprozess, Prozessdauer(h), Prozessort(x)(h), Prozessort(y)(h), frühester Starttag, vorläufiger Plantag, späteste Fälligkeit, Dienstleistungsname, Prozesswichtigkeit wp
services={
('0'): [('P1', 'r3', '-', 2.25, 0.25, 1.0, 3, 4, 5, 'Wartung', 0.8)],
('1'): [('P2', 'r3', 'P5', 0.75, 2.0, 1.0, 1, 2, 4, 'Spindelwechsel', 0.8)],
('2'): [('P3', 'r1', '-', 3.25, 1.75, 2.75, 3, 5, 5, 'Beratung', 0.8)],
('3'): [('P4', 'r3', '-', 1.0, 3.0, 3.0, 1, 1, 1, 'Reparatur', 1.0)],
('4'): [('P5', 'r1', '-', 1.75, 2.0, 1.0, 1, 3, 4, 'Lagerwechsel', 0.8)],
('5'): [('P6', 'r1', '-', 1.75, 2.25, 1.5, 5, 5, 5, 'Schulung', 0.8)],
('6'): [('P7', 'r3', '-', 3.75, 3.0, 2.0, 1,2,5, 'Inspektion', 0.5)],
('7'): [('P8', 'r3', 'P13', 0.75, 0, 2.0,2,2,4, 'Wartung', 0.8)],
('8'): [('P9', 'r1', '-', 3.0, 1.25, 1.75, 1,2,4, 'Optimierung', 0.8)],
('9'): [('P10', 'r3', '-', 3.0, 2.0, 0.25, 3,3,5, "Wartung", 0.8)],
('10'): [('P11', 'r1', '-', 3.0, 2.5, 3.0, 2,5,5, "Lagerwechsel", 0.8)],
('11'): [('P12', 'r2', '-', 2.5, 2.5, 1.75, 1,2,5, "Inspektion", 0.5)],
('12'): [('P13', 'r1', '-', 1.0, 0.0, 2.00, 2,2,4, "Wartung", 0.8)],
('13'): [('P14', 'r2', '-', 2.0, 3.0, 2.25, 1,2,2, "Wartung", 0.8)],
('14'): [('P15', 'r3', '-', 1.25, 1.0, 0.75, 2,4,5, "Inspektion", 0.5)],
('15'): [('P16', 'r1', '-', 0.5, 2.75, 2.5, 1,1,4, " Inspektion", 0.5)],
('16'): [('P17', 'r2', '-', 2.25, 1.5, 1.0, 1,2,4, "Wartung", 0.8)],
('17'): [('P18', 'r3', '-', 2.5, 2.0, 2.75, 1,3,5, "Inspektion", 0.5)],
('18'): [('P19', 'r1', '-', 0.75, 1.75, 1.75, 2,2,5, "Beratung", 0.8)],
('19'): [('P20', 'r3', 'P21', 0.75, 2.0, 0, 4,4,4, "Montage", 0.8)],
('20'): [('P21', 'r3', '-', 1.25, 3.0, 2.75, 4,4,4, "Teleservice", 1.0)],
('Origin'): [( "", "","" , 0, 0, 0)], # only prozessort
}
#('Origin'):[( "", "","" , 0, 0, 0, 1, 1, 1, "ssda", 1.0)], # only prozessort
#('-'):[('-', '-', '-', '-', '-', '-', '-','-', '-', '-')]}

#@inline

global V
V = 80

def coordinates(services):

    # global locations
    locations = distances = []

    [locations.append([V * services[str(i)][0][4], V * services[str(i)][0][5]]) for i in
     services]  # velocity * time = distance in km or m

    x, y = zip(* locations)
    plt.scatter(*zip(* locations))
    return locations

def euclidean_distances(loc):
    n = 0
    a = np.array(loc)
    b = a.reshape(a.shape[0], 1, a.shape[1])

    return np.sqrt(np.einsum('ijk, ijk->ij', a - b, a - b))

def worker_order_list(solution):
    # global r1, r2, r3

    r1 = []
    r2 = []
    r3 = []

    try:
        [r1.append(_) for _ in range(len(solution)) if services[str(_)][0][1] == 'r1']
        [r2.append(_) for _ in range(len(solution)) if services[str(_)][0][1] == 'r2']
        [r3.append(_) for _ in range(len(solution)) if services[str(_)][0][1] == 'r3']

        # else:print("there is an assignment problem")

    except:
        print("One of the worker entries in dictionary is wrong")

    return [r1, r2, r3]


def worker_day_list(r1, r2, r3, solution):
    # global r1_days, r2_days, r3_days

    r1_days = []
    r2_days = []
    r3_days = []

    [r1_days.append(solution[_]) for _ in r1]
    [r2_days.append(solution[_]) for _ in r2]
    [r3_days.append(solution[_]) for _ in r3]

    return [r1_days, r2_days, r3_days]


def assign_schedule(rx, rx_days):
    d1 = []
    d2 = []
    d3 = []
    d4 = []
    d5 = []

    try:

        d1 = [rx[_] for _ in range(len(rx_days)) if rx_days[_] == 0]
        d2 = [rx[_] for _ in range(len(rx_days)) if rx_days[_] == 1]
        d3 = [rx[_] for _ in range(len(rx_days)) if rx_days[_] == 2]
        d4 = [rx[_] for _ in range(len(rx_days)) if rx_days[_] == 3]
        d5 = [rx[_] for _ in range(len(rx_days)) if rx_days[_] == 4]

    except:
        print("one of the orders is scheduled on an invalid day, please check")

    # global r1_schedule, r2_schedule, r3_schedule
    rx_schedule = [d1, d2, d3, d4, d5]

    return rx_schedule


def travel_distance(rx_schedule):
    # global total_distance
    distanceToFirstOrder = 0
    daily_distances = []

    for day in range(len(rx_schedule)):
        total_distance = 0
        try:
            firstOrderinDay = rx_schedule[day][0]
            distanceToFirstOrder = distances_matrix[firstOrderinDay][21]  # initial distance to first order on day-x

            total_distance += distanceToFirstOrder

            # print("day", day,"\nfirst order is", firstOrderinDay,"and the distance to that order from center is", distanceToFirstOrder)
            # print("now total distance ist", total_distance)

            for order in range(len(rx_schedule[day])):

                try:
                    current = rx_schedule[day][order]  # current order number
                    next = rx_schedule[day][order + 1]
                    # print("distance between order", current, "and order", next, "is", distances_matrix[current][next])
                    total_distance += distances_matrix[current][next]  # update total_distance.
                    # print("now total distance ist", total_distance)

                    # put value of index after it ..

                except:
                    current = rx_schedule[day][order]  # current order number
                    LAST_ORDER = 21
                    # print("last order, distance between order", current, "and center", next, "is", distances_matrix[current][next])
                    total_distance += distances_matrix[current][LAST_ORDER]  # update total_distance.
                    # print("now total distance ist", total_distance)
            daily_distances.append(total_distance)
            #######print("\n")

        except:
            daily_distances.append(0)
            continue
            # print("for the day", day, "there are no orders for this worker, free day!")
            # print("now total distance ist", total_distance)

    # iterate over the daily schedule itself and add distances -------------------

    #####print("final total distance is", total_distance)

    total_d = sum(daily_distances)

    return total_d, daily_distances


def daily_time(rx_schedule):  # insert try loops cause it might be empty lists.

    daily_time_worker = []
    daily_service_time_worker = []
    daily_travel_time_worker = []

    daily_travel_time_worker = [x / V for x in travel_distance(rx_schedule)[1]]  # this is a list
    # print(daily_travel_time_worker)

    for _ in range(len(rx_schedule)):  # use try except blocks for empty lists.
        service_time = 0

        for _ in rx_schedule[_]:
            try:
                # print(j)
                service_time += services[str(_)][0][3]
            except:
                continue

        daily_service_time_worker.append(service_time)  # this is a list
        sum_tp= sum(daily_service_time_worker)
    # print(daily_service_time_worker)

    daily_time_worker = [x + y for x, y in zip(daily_service_time_worker, daily_travel_time_worker)]

    return [daily_time_worker, sum_tp]  # type list


def travel_cost(rx_schedule):
    TRAVEL_COST_HOUR = 60  # €/h;

    # we need to extract this variable from previous function but how?

    # for r1_schedule the total_distance
    # take this total_distance from each instance of the function, not only last value saved.
    travel_time = travel_distance(rx_schedule)[0] % V
    travel_cost = travel_time * (TRAVEL_COST_HOUR)  # here its the time multiplied by the travel cost per hour, which gives us total travelling cost in the week..

    # working_time=services[0][][]

    return travel_cost

def total_cost(rx, rx_schedule, Lohn):
    total_Cost = 0

    for _ in range(len(rx)):
        total_Cost += (services[str(rx[_])][0][3] * Lohn)  # this is cost of each of the orders. time* pay of the worker

    total_Cost += travel_cost(rx_schedule)

    # time elapsed in each order of the schedule

    return total_Cost

global R1_SALARY
global R2_SALARY
global R3_SALARY

R1_SALARY = 80
R2_SALARY = 60
R3_SALARY = 60


def fitness_cost(solution):

    r1 = worker_order_list(solution)[0]
    r2 = worker_order_list(solution)[1]
    r3 = worker_order_list(solution)[2]

    r1_days = worker_day_list(r1, r2, r3, solution)[0]
    r2_days = worker_day_list(r1, r2, r3, solution)[1]
    r3_days = worker_day_list(r1, r2, r3, solution)[2]

    r1_schedule = assign_schedule(r1, r1_days)
    r2_schedule = assign_schedule(r2, r2_days)
    r3_schedule = assign_schedule(r3, r3_days)

    r1_total_distance = travel_distance(r1_schedule)[0]
    r2_total_distance = travel_distance(r2_schedule)[0]
    r3_total_distance = travel_distance(r3_schedule)[0]

    r1_travel_cost = travel_cost(r1_schedule)
    r2_travel_cost = travel_cost(r2_schedule)
    r3_travel_cost = travel_cost(r3_schedule)

    r1_total_cost = total_cost(r1, r1_schedule, R1_SALARY)
    r2_total_cost = total_cost(r2, r2_schedule, R2_SALARY)
    r3_total_cost = total_cost(r3, r3_schedule, R3_SALARY)

    fitness = r1_total_cost + r2_total_cost + r3_total_cost

    return fitness

def oz(daily_time_rx):
  ol_m= 0; ot_r=0
  for _ in range(len(daily_time_rx)):
    if daily_time_rx[_] > 10:
      ot_r+= daily_time_rx[_]-10 # define OT zahl
      ol_m+= 1
  return ol_m, ot_r

def fitness_function(solution, c_min, c_max_op):  # formulas defined in Finke page 101/180. Seite.83

    list1 = [];
    list2 = []  ##----------turning solution array to new_list array- 2D

    for _ in range(len(solution)):
        list1.append(_)
        list2.append(solution[_])

    new_list = [(list1[_], list2[_]) for i in range(0, len(list1))]

    fitness = 0

    # problem preparation again ----------------------------------------- ######

    r1 = worker_order_list(solution)[0]
    r2 = worker_order_list(solution)[1]
    r3 = worker_order_list(solution)[2]

    r1_str = worker_order_list(str_plan)[0]
    r2_str = worker_order_list(str_plan)[1]
    r3_str = worker_order_list(str_plan)[2]

    r1_days = worker_day_list(r1, r2, r3, solution)[0]
    r2_days = worker_day_list(r1, r2, r3, solution)[1]
    r3_days = worker_day_list(r1, r2, r3, solution)[2]

    r1_days_str = worker_day_list(r1_str, r2_str, r3_str, str_plan)[0]
    r2_days_str = worker_day_list(r1_str, r2_str, r3_str, str_plan)[1]
    r3_days_str = worker_day_list(r1_str, r2_str, r3_str, str_plan)[2]

    r1_schedule = assign_schedule(r1, r1_days)
    r2_schedule = assign_schedule(r2, r2_days)
    r3_schedule = assign_schedule(r3, r3_days)

    r1_str_schedule = assign_schedule(r1_str, r1_days_str)
    r2_str_schedule = assign_schedule(r2_str, r2_days_str)
    r3_str_schedule = assign_schedule(r3_str, r3_days_str)

    r1_total_distance = travel_distance(r1_schedule)[0]
    r2_total_distance = travel_distance(r2_schedule)[0]
    r3_total_distance = travel_distance(r3_schedule)[0]

    r1_travel_cost = travel_cost(r1_schedule)
    r2_travel_cost = travel_cost(r2_schedule)
    r3_travel_cost = travel_cost(r3_schedule)

    r1_total_cost = total_cost(r1, r1_schedule, R1_SALARY)
    r2_total_cost = total_cost(r2, r2_schedule, R2_SALARY)
    r3_total_cost = total_cost(r3, r3_schedule, R3_SALARY)

    fitness = r1_total_cost + r2_total_cost + r3_total_cost

    print('fitness for individual in generation before kennzahl:{:5f}'.format(fitness))

    daily_time_r1 = daily_time(r1_schedule)[0]
    daily_time_r2 = daily_time(r2_schedule)[0]
    daily_time_r3 = daily_time(r3_schedule)[0]

    # ------------------------------ 1__Pünktlichkeitskennzahl ------------------------------------#

    z_sum = 0

    for _ in range(len(new_list)):

        order = new_list[_][0]
        est = services[str(order)][0][6]  # frühst möglichen Soll-Starttermin, frühester Starttag
        due = services[str(order)][0][8]  # soll End-termin, späteste Fälligkeit
        end = new_list[_][1]  # ist termin

        st = new_list[_][1]  # verfrühten Starttermin
        TH = 5  # planungshorizont

        def dl_op(est, st, end, due):  # Operativen Ressourcenplanung.. also Gewichtungsfaktor?

            if st < est:
                try:
                    return (est - st) / (2 * TH)  #careful, if due =est --- division by zero
                except ZeroDivisionError:
                    print("Error dividing by zero, if block")
            elif st >= est and end <= due:
                return 0
            else:
                return (end - due) / TH

        dl_op = dl_op(est, st, end, due)

        wt_op = TH / (TH + 2 * st)  # Gewichtung

        w = services[str(order)][0][10]  # prozesswichtigkeit from dictionary

        z_op = 1.0 - (dl_op * wt_op * w)  # Zeitgenauigkeit eines Prozesses    #return TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'

        z_sum += z_op
        # print(z_sum)

    n = len(new_list)
    zi = z_sum / n

    # print("z_avg", type(z_avg), z_avg)

    # ------------------------------   2___ Kostenkennzahl ------------------------------------

    # c_max_op: from genetic, c_min als Fixwert die Kosten der zugrunde liegenden strategischen Planung

    ci = fitness  # from actual Individual from population

    cei = (c_max_op - ci) / (c_max_op - c_min)
    print('ci', ci, 'c_max_op', c_max_op)
    print('c_max_op-ci', c_max_op - ci, 'c_max_op-c_min', c_max_op - c_min)
    print(cei)
    # if cei<0:
    # print("ce is negative", cei)
    # sys.exit()
    # print('cei:{:5f}'.format(cei))

 #TODO fix cei<0 issue

    # print("cei", cei, "ci", ci, "c_min", c_min)

    # ---------------------------------------   3___ Auslastungskennzahl ------------------------------------# Randbedingung, dass Ressourcenüberlastungen unzulässig sind, abzubilden.

    # ol - Überlastungskennzahl-------------------massiver Überlastung einzelner Ressourcen physikalisch / rechtlich unmöglich ist

    # 1. menschliche Ressourcen r1, r2 usw ..... r_m

    ol_m_total = oz(daily_time_r1)[0] + oz(daily_time_r2)[0] + oz(daily_time_r3)[0]  # [0] is the ol_m
    oli = math.pow(0.1, ol_m_total) #always positive

    # 2. technische P1, P2 usw ..

    # ----------------------------------------------------- ot - Überstunden Kennzahl für menschliche Ressourcen.

    ot_r= oz(daily_time_r1)[1] + oz(daily_time_r2)[1] + oz(daily_time_r3)[1]  # sum of all schedules overtime schedules of all workers, #[1] is the ot_r

    OT_MAX_R= 30 #taken from page 163

    ###################
    try:
        oti = 1.0 - (ot_r/OT_MAX_R) * 0.9  #floating point  #always positive
    except:
        oti =   1


#tend-tst

    def ai_rj(rx_schedule):
        ai= 0
        sum_tp = 0
        sum_diff = 0

        for _ in rx_schedule:
            for _ in _:
                try:
                    tst= services[str(_)][0][6]
                    top= services[str(_)][0][3] #time in hours
                    tend= services[str(_)][0][8]
                    sum_tp+= top
                    sum_diff+= (tend-tst)* 8 #todo should this time be in hours too? multiply by what? 8, 10?
                except:
                    continue
        return [sum_tp,sum_diff]

    sum_tp_r1 = ai_rj(r1_schedule)[0]
    sum_tp_r2 = ai_rj(r2_schedule)[0]
    sum_tp_r3 = ai_rj(r3_schedule)[0]

    sum_diff_r1 = ai_rj(r1_schedule)[1]
    sum_diff_r2 = ai_rj(r2_schedule)[1]
    sum_diff_r3 = ai_rj(r3_schedule)[1]

    #sum of all divided by each other, page 106

    all_sum_tp = sum_tp_r1 + sum_tp_r2 + sum_tp_r3
    all_sum_diff = sum_diff_r1 + sum_diff_r2 + sum_diff_r3


    ai_op= all_sum_tp/ all_sum_diff * oli * oti

    if ai_op > 1:     # todo ai_op < 1 immer
        print("aM is bigger than 1")
        sys.exit()

    #    sum_tp= daily_time(r1_schedule)[1] + daily_time(r2_schedule)[1] + daily_time(r3_schedule)[1]     #nominator
    #    sum_t_diff = 0
    #   for i in range(len(solution)):
    #      a = solution[i] - services[str(i)][0][7]
    #    sum_t_diff += a
    # print(sum_t_diff)

    # ------------------------------ Kennzahl-Gewichtung ------------------------------------------------------#

    # Dabei gibt der Gewichtungsfaktor wz die Priorität der Kennzahl z an, wobei wz = 1 bedeutet,
    # dass diese Kennzahl direkt in die Berechnung der Gesamtfitness eingeht, bei wz = 0
    # wird diese Kennzahl nicht berücksichtigt, und 0 < wz < 1 gibt eine entsprechende Abschwächung der Kennzahl an

    # total must be 1, Gewichtungsfaktoren vom Kennzahlen

    wz = 0.5
    wa = 0.3
    wc = 0.2

    zw  = zi  + ((1 -  wz) * (1 - zi))
    aw  = ai_op  + ((1 -  wa) * (1 - ai_op))
    cew = cei + ((1 -  wc) * (1 - cei))

    # ------------------- Operational Gesamt Fitness -------------------------------------
    f_op = fitness * zw * cew  * aw
    print("f_op after Kennzahl", f_op)

    def start_time(p, rx_schedule):
        for _ in rx_schedule:
            time = 0
            for _ in _:
                try:
                    if _ == p:
                        time = 0
                        return time
                    else:
                        continue
                except:
                    continue



    # ----------------------komplexitätskennzahl
    def xz(solution):
        NP = 20       #number of processes
        for _ in range(len(solution)):
        #solution
            jp = solution[_] #geplanter Tag
            p = _
            if _ in r1:
                stp = start_time(p,r1_schedule)
            if _ in r2:
                stp = start_time(p, r2_schedule)
            if _ in r3:
                stp = start_time(p, r3_schedule)

                #strategic planning

            jp0 = str_plan[_]

            if _ in r1_str:
                stp0 = start_time(p, r1_str_schedule)
            if _ in r2_str:
                stp0 = start_time(p, r2_str_schedule)
            if _ in r3_str:
                stp0 = start_time(p, r3_str_schedule)


            if jp == jp0 and stp == stp0:
                const = 0
            if jp == jp0 and stp!=stp0:
                const = 0.5
            if jp != jp0:
                const = 1

            const+= const
        return (NP-const) / NP

    # Alternativ Prozessem Komplexität

    #def ap_max():
     #   for i in range(len(solution)):
      #      if

    # Alternativ Ressourcen Komplexität


    xz_result = xz(solution)

    f_op_corr = f_op * xz_result

    return f_op_corr

   # ----------------------------------------------------------------------------------#-----------------------------------------------------------------#----------------------------


domain = [(0,4)] * (len(services) - 1)

# TODO make a better random search algorithm

def random_search(domain, fitness_function):
  best_cost = sys.maxsize
  for _ in range(1000):
    solution = [random.randint(domain[_][0],domain[_][1]) for _ in range(len(domain))]
    cost = fitness_function(solution)
    if cost < best_cost:
      best_cost = cost
      best_solution = solution
  return best_solution

def mutation(domain, step, solution):     #here we choose the random schedule
  gene = random.randint(0, len(domain) - 1)
  mutant = solution
  if random.random() < 0.5:
    if solution[gene] != domain[gene][0] and (solution[gene] - step) >= 0:
      mutant = solution[0:gene] + [solution[gene] - step] + solution[gene + 1:]
    else:
      mutation(domain, step, solution)
  else:
    if solution[gene] != domain[gene][1] and (solution[gene] + step) <= 4:
      mutant = solution[0:gene] + [solution[gene] + step] + solution[gene + 1:]
    else:
      mutation(domain, step, solution)
  return mutant

def crossover(domain, solution1, solution2):
  gene = random.randint(1, len(domain) - 2)   #where i want to cut the chromosomes
  return solution1[0:gene] + solution2[gene:]

#todo   mutationswahrscheinlichkeit, what to do with it in book?
def genetic(domain, fitness_function, population_size=5, step=1,  ##check input arguments of fitness_function
            probability_mutation=0.3, elitism=0.2,
            number_generations=5, search=False):
    best_individual_in_generation = []  ## for vibrating plot
    generation = []
    ######################

    population = []
    for _ in range(population_size):
        if search == True:
            solution = random_search(domain, fitness_function)  ### can i make a better randomization function ??
        else:
            solution = [random.randint(domain[_][0], domain[_][1]) for i in range(len(domain))]

        population.append(solution)

    print("population", population)  # checked nothing wrong

    # global c_max_op       ### why is it always equal to c_min ??

    c = [(fitness_cost(individual)) for individual in population]
    # c = [(fitness_function(individual)) for individual in population]
    # print(c_max)
    c.sort()
    c_max_op = c[-1]  # take the last element of the list (cmax_op wird dynamisch berechnet und pro Population mit den Kosten des jeweils teuersten Individuums belegt
    print("c_max_op", c_max_op)

    number_elitism = int(elitism * population_size)  # this is the number of individuals we will take from the population.
    costs = []
    for i in range(number_generations):
        costs = [(fitness_function(individual, c_min, c_max_op), individual) for individual in population]  # define costs array of fitness and individuals in the population of 20 solutions.
        costs.sort()                #individual was in costs too
        print("costs", costs)

        print("best fitness for individual in generation", costs[0][0])  # this code shows that the fitness is decreasing, GA works, but we need to add constraints

        ###############################
        best_individual_in_generation.append(costs[0][0])  ## these 2 lines for vibration plot.
        generation.append(_)
        ####################################################
        ordered_individuals = [individual for (cost, individual) in costs]  ###  individuals but orderened array.

        population = ordered_individuals[0:number_elitism]  # extract the ordered individuals: first number_elitism of the population
        while len(population) < population_size:
            if random.random() < probability_mutation:
                m = random.randint(0, number_elitism)  # chooses which chromosome i should mutate
                population.append(mutation(domain, step, ordered_individuals[m]))
            else:
                i1 = random.randint(0, number_elitism)
                i2 = random.randint(0, number_elitism)
                population.append(crossover(domain, ordered_individuals[i1], ordered_individuals[i2]))

    # global genetic_solution

    #### plotting code of convergence

    # plt.plot(generation, best_individual_in_generation, label= "fitness")
    # plt.xlabel("generation number")
    # plt.ylabel("fitness_cost")
    # plt.title("Convergence of fitness cost through generations")
    # plt.legend()
    # plt.show()

    return [costs[0][0], costs[0][1]]
    # , generation, best_individual_in_generation]

def loop_genetic(domain, fitness_function):
  a = 6000
  for _ in range(3):
    b = genetic(domain, fitness_function)[0]
    c = genetic(domain, fitness_function)[1]

    # TODO: here we will define plt plots in the loop

    plt.plot(genetic(domain, fitness_function)[2], genetic(domain, fitness_function)[3], label= "fitness")
    plt.xlabel("generation number")
    plt.ylabel("fitness_cost")
    plt.title("Convergence of fitness cost through generations")
    plt.legend()
    plt.show()


    if b < a:
      a = b
      final_solution = c
    else: continue

  return [a, final_solution]


str_plan = [(services[str(i)][0][7]) for i in range(len(services)-1)]    #vorläufiger Plantag
print(str_plan)
c_min = fitness_cost(str_plan)    # fitness cost of strategic planning
print('c_min:{:5f}'.format(c_min))

locations = coordinates(services)
distances_matrix = euclidean_distances(locations)
distances_matrix

solution_best = genetic(domain, fitness_function)[1]


def print_function(solution):  # why is my function not changing parameters ????

    r1 = worker_order_list(solution)[0]
    r2 = worker_order_list(solution)[1]
    r3 = worker_order_list(solution)[2]

    r1_days = worker_day_list(r1, r2, r3, solution)[0]
    r2_days = worker_day_list(r1, r2, r3, solution)[1]
    r3_days = worker_day_list(r1, r2, r3, solution)[2]

    r1_schedule = assign_schedule(r1, r1_days)
    r2_schedule = assign_schedule(r2, r2_days)
    r3_schedule = assign_schedule(r3, r3_days)

    schedules = ["r1_schedule", r1_schedule, "r2_schedule", r2_schedule, "r3_schedule", r3_schedule]

    #  for i in schedules:
    #   print(i)

    return schedules

    ##function is taking old values and results why??

a = print_function(solution_best)
print(a)


print("--- %s seconds ---" % (time.time() - start_time))


