import time
import random
import math
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from datetime import timedelta
import csv
from functools import cache, lru_cache     ##this is caching decorators we can directly use
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

############################### Caching to make code run faster ###################################


#cache = {}

def cache_wrapper(func):
    # Add a dictionary for `func` in the cache
    if not func in cache:
        cache[func] = {}

    # Create a new cached function
    def cached_func(*args):
        # If entry is cached, return from cache
        if args in cache[func]:
            return cache[func][args]

        # If not, add it to cache and return it
        value = func(*args)
        cache[func][args] = value
        return value

    # Return our new cached function
    return cached_func






global V
V = 80

######## using a decorator to cache here #######

#@cache_wrapper
#@cache
def coordinates(services):
    """

    :param services:
    :type services: dictionary
    :return: locations
    :rtype: array
    """
    # global locations
    locations = distances = []

    [locations.append([V * services[str(i)][0][4], V * services[str(i)][0][5]]) for i in
     services]  # velocity * time = distance in km or m

    x, y = zip(* locations)
    plt.scatter(*zip(* locations))
    return locations


def euclidean_distances(loc):
    """

    :param loc: location
    :type loc: coordinates
    :return: distances
    :rtype: distance (meter)
    """
    n = 0
    a = np.array(loc)
    b = a.reshape(a.shape[0], 1, a.shape[1])

    return np.sqrt(np.einsum('ijk, ijk->ij', a - b, a - b))

###### Using a simple function call. ######
#euclidean_distances(loc) = cache_wrapper(euclidean_distances(loc))

#@lru_cache
def worker_order_list(solution):
    """

    :param solution:
    :type solution:
    :return:
    :rtype:
    """
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
    """

    :param r1:
    :type r1:
    :param r2:
    :type r2:
    :param r3:
    :type r3:
    :param solution:
    :type solution:
    :return:
    :rtype:
    """

    r1_days = []
    r2_days = []
    r3_days = []

    [r1_days.append(solution[_]) for _ in r1]
    [r2_days.append(solution[_]) for _ in r2]
    [r3_days.append(solution[_]) for _ in r3]

    return [r1_days, r2_days, r3_days]


def assign_schedule(rx, rx_days):
    """

    :param rx:
    :type rx:
    :param rx_days:
    :type rx_days:
    :return:
    :rtype:
    """
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


def travel_distance(rx_schedule, distanceToFirstOrder = 0, LAST_ORDER = 21):
    """

    :param rx_schedule:
    :type rx_schedule:
    :param distanceToFirstOrder:
    :type distanceToFirstOrder:
    :param LAST_ORDER:
    :type LAST_ORDER:
    :return:
    :rtype:
    """
    # global total_distance
    daily_distances = []

    ## Gantt Chart
    ganttChartTimesDay = []
    ## Gantt Chart

    for day in range(len(rx_schedule)):
        total_distance = 0

        ## Gantt Chart
        firstOrderTime = 8  ## stunden
        startTimesDay = []
        ## Gantt Chart

        try:
            firstOrderinDay = rx_schedule[day][0]
            distanceToFirstOrder = distances_matrix[firstOrderinDay][21]  # initial distance to first order on day-x

            ##### Gantt Chart

            firstOrderTime = 8 + distanceToFirstOrder / V
            startTimesDay.append(firstOrderTime)

            ###### Gantt Chart

            total_distance += distanceToFirstOrder

            # print("day", day,"\nfirst order is", firstOrderinDay,"and the distance to that order from center is", distanceToFirstOrder)
            # print("now total distance ist", total_distance)

            nextOrderStartTime = firstOrderTime

            for order in range(len(rx_schedule[day])):

                try:
                    current = rx_schedule[day][order]  # current order number
                    next = rx_schedule[day][order + 1]
                    # print("distance between order", current, "and order", next, "is", distances_matrix[current][next])
                    total_distance += distances_matrix[current][next]  # update total_distance.
                    # print("now total distance ist", total_distance)

                    ## Gantt Chart

                    nextOrderStartTime += distances_matrix[current][next] / V + services[str(order)][0][3]
                    startTimesDay.append(nextOrderStartTime)

                    #### Gantt Chart

                    # put value of index after it ..

                except:
                    current = rx_schedule[day][order]  # current order number
                    # print("last order, distance between order", current, "and center", next, "is", distances_matrix[current][next])
                    total_distance += distances_matrix[current][LAST_ORDER]  # update total_distance.
                    # print("now total distance ist", total_distance)

                    ## Gantt Chart

                    ganttChartTimesDay.append(startTimesDay)

                    ## Gantt Chart

            daily_distances.append(total_distance)
            #######print("\n")

        except:
            daily_distances.append(0)

            ## Gantt Chart
            ganttChartTimesDay.append([])      # no orders in the day
            ## Gantt Chart

            continue
            # print("for the day", day, "there are no orders for this worker, free day!")
            # print("now total distance ist", total_distance)

    # iterate over the daily schedule itself and add distances -------------------

    #####print("final total distance is", total_distance)

    total_d = sum(daily_distances)

    return total_d, daily_distances, ganttChartTimesDay


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


def travel_cost(rx_schedule, TRAVEL_COST_HOUR = 60):     # €/h;

    # we need to extract this variable from previous function but how?

    # for r1_schedule the total_distance
    # take this total_distance from each instance of the function, not only last value saved.
    travel_time = travel_distance(rx_schedule)[0] % V
    travel_cost = travel_time * (TRAVEL_COST_HOUR)  # here its the time multiplied by the travel cost per hour, which gives us total travelling cost in the week..

    # working_time=services[0][][]

    return travel_cost

def total_cost(rx, rx_schedule, Lohn, total_Cost = 0):
    for _ in range(len(rx)):
        total_Cost += (services[str(rx[_])][0][3] * Lohn)  # this is cost of each of the orders. time* pay of the worker

    total_Cost += travel_cost(rx_schedule)

    # time elapsed in each order of the schedule
    return total_Cost

global CONST_R1_SALARY
global CONST_R2_SALARY
global CONST_R3_SALARY

CONST_R1_SALARY = 80
CONST_R2_SALARY = 60
CONST_R3_SALARY = 60


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

    r1_total_cost = total_cost(r1, r1_schedule, CONST_R1_SALARY)
    r2_total_cost = total_cost(r2, r2_schedule, CONST_R2_SALARY)
    r3_total_cost = total_cost(r3, r3_schedule, CONST_R3_SALARY)

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

    r1_total_cost = total_cost(r1, r1_schedule, CONST_R1_SALARY)
    r2_total_cost = total_cost(r2, r2_schedule, CONST_R2_SALARY)
    r3_total_cost = total_cost(r3, r3_schedule, CONST_R3_SALARY)

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
        CONST_TH = 5  # planungshorizont

        def dl_op(est, st, end, due):  # Operativen Ressourcenplanung.. also Gewichtungsfaktor?

            if st < est:
                try:
                    return (est - st) / (2 * CONST_TH)  #careful, if due =est --- division by zero
                except ZeroDivisionError:
                    print("Error dividing by zero, if block")
            elif st >= est and end <= due:
                return 0
            else:
                return (end - due) / CONST_TH

        dl_op = dl_op(est, st, end, due)

        wt_op = CONST_TH / (CONST_TH + 2 * st)  # Gewichtung

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

    CONST_wz = 0.5
    CONST_wa = 0.3
    CONST_wc = 0.2

    zw  = zi  + ((1 -  CONST_wz) * (1 - zi))
    aw  = ai_op  + ((1 -  CONST_wa) * (1 - ai_op))
    cew = cei + ((1 -  CONST_wc) * (1 - cei))

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


######################################  Dictionary Entries ####################################

# 1- worker id (Resource)
# 2- Task Number (Order number from schedule)

##### START and FINISH of Orders ######

# 3- START times of service orders (timeline format).
# 4- FINISH times of service orders (timeline format).

####### - START AND FINISH OF Travel times represented by the gaps between end of an order and start of next order. #######

# 5- START of next service order (timeline format).
# 6- FINISH of previous service order (timeline format).

def serviceTimes(rx_schedule):
    serviceTimesArray = []
    for day in rx_schedule:
        dayServiceTimes = []
        for order in day:
            dayServiceTimes.append(services[str(order)][0][3])
        serviceTimesArray.append(dayServiceTimes)

    return serviceTimesArray

def print_function(solution):

    r1 = worker_order_list(solution)[0]
    r2 = worker_order_list(solution)[1]
    r3 = worker_order_list(solution)[2]

    r1_days = worker_day_list(r1, r2, r3, solution)[0]
    r2_days = worker_day_list(r1, r2, r3, solution)[1]
    r3_days = worker_day_list(r1, r2, r3, solution)[2]

    r1_schedule = assign_schedule(r1, r1_days)
    r2_schedule = assign_schedule(r2, r2_days)
    r3_schedule = assign_schedule(r3, r3_days)

    schedules = [r1_schedule, r2_schedule, r3_schedule]

    ############################################ Gantt Chart #####################################

    startTimes_R1 = travel_distance(r1_schedule, distanceToFirstOrder=0, LAST_ORDER=21)[2]
    startTimes_R2 = travel_distance(r2_schedule, distanceToFirstOrder=0, LAST_ORDER=21)[2]
    startTimes_R3 = travel_distance(r3_schedule, distanceToFirstOrder=0, LAST_ORDER=21)[2]

    serviceTimes(r1_schedule)
    serviceTimes(r2_schedule)
    serviceTimes(r3_schedule)

    ################### End times - worker R1 ########################
    endTimes_R1 = []

    for i in range(len(r1_schedule)):
        endTimes = []
        for x in range(len(r1_schedule[i])):
            endTimes.append(startTimes_R1[i][x] + serviceTimes(r1_schedule)[i][x])
        endTimes_R1.append(endTimes)

    ################### End times - worker R2 ########################
    endTimes_R2 = []

    for i in range(len(r2_schedule)):
        endTimes = []
        for x in range(len(r2_schedule[i])):
            endTimes.append(startTimes_R2[i][x] + serviceTimes(r2_schedule)[i][x])
        endTimes_R2.append(endTimes)

    ################### End times - worker R3 ########################
    endTimes_R3 = []

    for i in range(len(r3_schedule)):
        endTimes = []
        for x in range(len(r3_schedule[i])):
            endTimes.append(startTimes_R3[i][x] + serviceTimes(r3_schedule)[i][x])
        endTimes_R3.append(endTimes)


    ############################### Gantt Chart ####################

    return schedules, startTimes_R1, startTimes_R2, startTimes_R3, endTimes_R1, endTimes_R2, endTimes_R3


###### How to Convert Integer to Datetime in Pandas DataFrame? ##################

############# Convert all integer Times to Datetime in PANDAS: from dataframes to "to_dict" ###################


#print("--- %s seconds ---" % (time.time() - start_time))

###################################### make for loop to put all arrays of 5 arrays in 1 list only. #####################################

returnPrintFunction = print_function(solution_best)

r1_schedule= returnPrintFunction[0][0]
r2_schedule= returnPrintFunction[0][1]
r3_schedule= returnPrintFunction[0][2]

############################### r1_List , r1_Days_List #########################

# Define days 0-4 sa dates...

r1_List = []
r1_Days_List= []

for i in range(len(r1_schedule)):
    for x in range(len(r1_schedule[i])):
        r1_List.append(r1_schedule[i][x])
        r1_Days_List.append(solution_best[r1_schedule[i][x]])

r1_List =  [str(int) for int in r1_List]
r1_Days_List = [str(int) for int in r1_Days_List]

string = 'Order'
r1ListDict = list(map(lambda orig_string: string + ' ' + orig_string, r1_List))

print("r1ListDict", r1ListDict)

############################### r2_List , r2_Days_List #########################

r2_List = []
r2_Days_List= []

for i in range(len(r2_schedule)):
    for x in range(len(r2_schedule[i])):
        r2_List.append(r2_schedule[i][x])
        r2_Days_List.append(solution_best[r2_schedule[i][x]])

r2_List =  [str(int) for int in r2_List]
r2_Days_List = [str(int) for int in r2_Days_List]

string = 'Order'
r2ListDict = list(map(lambda orig_string: string + ' ' + orig_string, r2_List))

print("r2ListDict", r2ListDict)

############################### r3_List , r3_Days_List #########################

r3_List = []
r3_Days_List= []

for i in range(len(r3_schedule)):
    for x in range(len(r3_schedule[i])):
        r3_List.append(r3_schedule[i][x])
        r3_Days_List.append(solution_best[r3_schedule[i][x]])

r3_List =  [str(int) for int in r3_List]
r3_Days_List = [str(int) for int in r3_Days_List]

string = 'Order'
r3ListDict = list(map(lambda orig_string: string + ' ' + orig_string, r3_List))

print("r3ListDict", r3ListDict)

############################ day Code to dateTime #########################################

def dateTimeDaysList(rx_Days_List):
    for i in range(len(rx_Days_List)):
        if rx_Days_List[i] == "0":
            rx_Days_List[i] = "2020-04-06"
        if rx_Days_List[i] == "1":
            rx_Days_List[i] = "2020-04-07"
        if rx_Days_List[i] == "2":
            rx_Days_List[i] = "2020-04-08"
        if rx_Days_List[i] == "3":
            rx_Days_List[i] = "2020-04-09"
        if rx_Days_List[i] == "4":
            rx_Days_List[i] = "2020-04-10"
    return rx_Days_List

dateTimeDaysListR1 = dateTimeDaysList(r1_Days_List)
dateTimeDaysListR2 = dateTimeDaysList(r2_Days_List)
dateTimeDaysListR3 = dateTimeDaysList(r3_Days_List)

############################ startTimes_Rx / EndTimes_Rx ##################################

startTimes_R1 = returnPrintFunction[1]
startTimes_R2 = returnPrintFunction[2]
startTimes_R3 = returnPrintFunction[3]

endTimes_R1 = returnPrintFunction[4]
endTimes_R2 = returnPrintFunction[5]
endTimes_R3 = returnPrintFunction[6]

############################## startTimesRx / endTimesRx ############################

def times(times_Rx):
    timesRx = []
    for i in range(len(times_Rx)):
        for x in range(len(times_Rx[i])):
            timesRx.append(times_Rx[i][x])
    return timesRx

startTimesR1 = times(startTimes_R1)
startTimesR2 = times(startTimes_R2)
startTimesR3 = times(startTimes_R3)

endTimesR1 = times(endTimes_R1)
endTimesR2 = times(endTimes_R2)
endTimesR3 = times(endTimes_R3)


###################################### dateTimeStartTimesRx / dateTimeEndTimesRx  ############################################

def dateTime(timesRx):
    dateTimeRx = []

    for time in timesRx:
        result = '{0:02.0f}:{1:02.0f}'.format(*divmod(time * 60, 60))
        dateTimeRx.append(str(result))

    return dateTimeRx

dateTimeStartTimesR1 = dateTime(startTimesR1)
dateTimeStartTimesR2 = dateTime(startTimesR2)
dateTimeStartTimesR3 = dateTime(startTimesR3)

dateTimeEndTimesR1 = dateTime(endTimesR1)
dateTimeEndTimesR2 = dateTime(endTimesR2)
dateTimeEndTimesR3 = dateTime(endTimesR3)


######## dateTimeOrderStartTimeRx for Gantt Chart (concatenate from dateTimeDaysListRx + dateTimeStartTimesRx) ##########

dateTimeOrderStartTimeR1 = [dateTimeDaysListR1[i] + " " + dateTimeStartTimesR1[i] for i in range(len(dateTimeDaysListR1))]
dateTimeOrderStartTimeR2 = [dateTimeDaysListR2[i] + " " + dateTimeStartTimesR2[i] for i in range(len(dateTimeDaysListR2))]
dateTimeOrderStartTimeR3 = [dateTimeDaysListR3[i] + " " + dateTimeStartTimesR3[i] for i in range(len(dateTimeDaysListR3))]

print("dateTimeOrderStartTimeR1", dateTimeOrderStartTimeR1)
print("dateTimeOrderStartTimeR2", dateTimeOrderStartTimeR2)
print("dateTimeOrderStartTimeR3", dateTimeOrderStartTimeR3)

########## dateTimeOrderEndTimeRx for Gantt Chart (concatenate from dateTimeDaysListR1 + dateTimeEndTimesR1) #############

dateTimeOrderEndTimeR1 = [dateTimeDaysListR1[i] + " " + dateTimeEndTimesR1[i] for i in range(len(dateTimeDaysListR1))]
dateTimeOrderEndTimeR2 = [dateTimeDaysListR2[i] + " " + dateTimeEndTimesR2[i] for i in range(len(dateTimeDaysListR2))]
dateTimeOrderEndTimeR3 = [dateTimeDaysListR3[i] + " " + dateTimeEndTimesR3[i] for i in range(len(dateTimeDaysListR3))]

print("dateTimeOrderEndTimeR1", dateTimeOrderEndTimeR1)
print("dateTimeOrderEndTimeR2", dateTimeOrderEndTimeR2)
print("dateTimeOrderEndTimeR3", dateTimeOrderEndTimeR3)

######################## RESOURCE LIST ############################

#TODO we can remove it because its a constant and can be directly entered in Resource of dict: Gantt Chart

r1List = ["R1" for i in range(len(r1ListDict))]
r2List = ["R2" for i in range(len(r2ListDict))]
r3List = ["R3" for i in range(len(r3ListDict))]

print("r1List", r1List)
print("r2List", r2List)
print("r3List", r3List)

#####################---------------------------- GANTT CHART ------------------------------###############
# components:
# 1- rxListDict
# 2- rxList
# 3- dateTimeOrderStartTimeRx
# 4- dateTimeOrderEndTimeRx

######################### Function for dict entries in each worker Gantt Chart. #########################



###################################### Worker R1 ###################################################

dict_R1 = ""
dict_R2 = ""
dict_R3 = ""

#def Dictionary_Rx(dict_Rx, rxListDict, dateTimeOrderStartTimeRx, dateTimeOrderEndTimeRx, Rx):
for i in range(len(r1ListDict)):
    dict_R1 += "dict(Task = " + '\"'+ r1ListDict[i] + '\"' + ", Start = " + '\"'+ dateTimeOrderStartTimeR1[i] + '\"' + ", Finish = " + '\"' +  dateTimeOrderEndTimeR1[i] + '\"' + \
    ", Resource= " + '\"'+ 'R1' +'\"' + ")" + ",\n"
dict_R1 = dict_R1[:-2]
#return dict_Rx

for i in range(len(r2ListDict)):
    dict_R2 += "dict(Task = " + '\"'+ r2ListDict[i] + '\"' + ", Start = " + '\"'+  dateTimeOrderStartTimeR2[i] + '\"' + ", Finish = " + '\"' + dateTimeOrderEndTimeR2[i] + '\"' + \
    ", Resource= " + '\"'+ 'R2' + '\"' + ")" + ",\n"
dict_R2 = dict_R2[:-2]

for i in range(len(r3ListDict)):
    dict_R3 += "dict(Task = " + '\"'+ r3ListDict[i] +'\"'+ ", Start = " + '\"'+ dateTimeOrderStartTimeR3[i] +'\"'+ ", Finish = " + '\"'+ dateTimeOrderEndTimeR3[i] + '\"' + \
    ", Resource= " + '\"'+ 'R3' + '\"' + ")" + ",\n"
dict_R3 = dict_R3[:-2]

#dict_R1 = Dictionary_Rx(dict_R1, r1ListDict, dateTimeOrderStartTimeR1, dateTimeOrderEndTimeR1, "'R1'")
#dict_R2 = Dictionary_Rx(dict_R2, r2ListDict, dateTimeOrderStartTimeR2, dateTimeOrderEndTimeR2, "'R2'")
#dict_R3 = Dictionary_Rx(dict_R3, r3ListDict, dateTimeOrderStartTimeR3, dateTimeOrderEndTimeR3, "'R3'")

print("dict_R1", dict_R1)
print("dict_R2", dict_R2)
print("dict_R3", dict_R3)