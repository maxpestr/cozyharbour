from typing import List
import logging
import networkx as nx
import itertools
from scipy.spatial import ConvexHull

from log_ui4 import *
from voronoi_clust8 import *

from merging import merge_clusters, merge_clusters_new

# параметры для интеллектуального поиска

low_param = 0.75
high_param = 1.25
num_of_intervals = 2

# раз в столько итераций будет рисовать. если поставить много, 1000000000000000 то ток старт и энд
kartinka_num = 10

# Point in R^n space
class Point:
    def __init__(self, id, cluster_id, coords):
        self.index = [id, cluster_id]
        self.coords = coords

    def __str__(self):
        return str(self.index)

# Main class for the problem
class Table:
    def __init__(self, points: List[Point]):
        self.points = points
        n = len(points)
        self.table = np.zeros(n ** 2).reshape(n, n)  # distance matrix
        self.clusters = {}
        self.maxclustid=-1
        self.tf=0.0
        self.edge_indexs={}
        self.triangulation=[]
        self.all_boundary_points={}

        self.tf=0 # основное target function, число, которое хотим чтобы росло
        ind=0  # забыл нахрена это, мож выкинуть?
        # словари номер в списке точек : айдишник и обратный
        self.idx_map = {i: point.index[0] for i, point in enumerate(self.points)}
        self.rev_idx_map = {point.index[0]: i for i, point in enumerate(self.points)}
        self.edgemap={}
        self.wGraph = None
        self.g_option=False # юзерская опция, если включена - граф берём из файла, а не строим по вороному.
        self.hull_points = self.convex_h() #точк выпуклой оболочки
        self.vor_for_image = self.get_vor_for_image() 
        self.check_geom_option = False # опция, надо ли проверять некое геом. условие

        # всякие отдельные множители tf за которыми юзер следит, если хочет, чтобы тоже не сильно падали
        self.additional_term=None
        self.omega_term = None
        self.bpn_term = None
        self.near_term = None
        self.fath_term = None
        self.clastnum=0
        self.po=False

    # для слежения за множителями tf
    def check_tf_is_better(self,at_max, ot_max, bpnt_max, neart_max, fatht_max, a_th=0.9, o_th=0.9,b_th=0.9,n_th=0.9,f_th=0.9):
        if self.additional_term < at_max*a_th:
            return 0
        if self.omega_term < ot_max*o_th:
            return 0
        if self.bpn_term < bpnt_max * b_th:
            return 0
        if self.near_term < neart_max * n_th:
            return 0
        if self.fath_term < fatht_max * f_th:
            return 0
        return 1

    # в случае 2d вороного для картинки строит
    def get_vor_for_image(self):
        if len(self.points[0].coords) == 2:
            all_points=[]
            clusters = {clust_id: [self.points[self.rev_idx_map[i]].coords for i in indices] for clust_id, indices in self.clusters.items()}
            for clust_id, points in clusters.items():
                all_points.extend(points)

            all_points.extend([[N_MAX, N_MAX], [-N_MAX, N_MAX], [N_MAX, -N_MAX], [-N_MAX, -N_MAX]])
            vor = Voronoi(np.array(all_points))
            return vor
        else:
            return None
    
    def convex_h(self):
        # Преобразуем набор точек в массив координат
        coords = np.array([point.coords for point in self.points])

        # Выполняем алгоритм Graham's scan для построения выпуклой оболочки
        hull = ConvexHull(coords)
        hull_points = set([self.points[i] for i in hull.vertices])

        return hull_points

    # итоговый датафрейм делает
    def process_dataframe(self, df, fname):
        # Добавляем колонки в датафрейм
        df['ID'] = [point.index[0] for point in self.points]
        df['Clust_ID'] = [point.index[1] for point in self.points]
        df['is_boundary?'] = [1 if point in self.all_boundary_points else 0 for point in self.points]
        df['is_in_convex_hull'] = [1 if point in self.hull_points else 0 for point in self.points]
        df['IDs_check_geom_points'] = [None] * len(self.points)
        df['distances_cgp'] = [None] * len(self.points)

        for i, point in enumerate(self.points):
            boundary_points, _ = find_boundary_points_for_id(point.index[0], self.points, self.rev_idx_map,self.edgemap)
            if len(boundary_points) > 0:
                bp_distances_and_points = [(self.dist(point, boundary_point), boundary_point) for boundary_point in boundary_points]
                bp_distances, bp_points = zip(*bp_distances_and_points)
                bp_check_results = [(boundary_point, self.check_geom_condition(boundary_point, point, boundary_points)) for boundary_point in boundary_points]
                bp_checked_points, bp_checked_distances = zip(*[(bp, dist) for bp, dist in bp_check_results if dist])

                bp_distance = [self.dist(point, boundary_point) for boundary_point in boundary_points]
                bp_nearest = [bp for bp, bp_dist in zip(boundary_points, bp_distance) if bp_dist==min(bp_distance)][0]
                df.at[i, 'ID_nearest_boundary_point'] = bp_nearest.index[0]
                df.at[i, 'Clust_ID_nearest_boundary_point'] = bp_nearest.index[1]

                df.at[i, 'IDs_check_geom_points'] = [bp.index[0] for bp in bp_points if bp in bp_checked_points]
                df.at[i, 'distances_cgp'] = [format(self.dist(point, bp), '.6f') for bp in bp_points if bp in bp_checked_points]

        # Сохраняем в xlsx
        df.to_excel(f'processed_{fname}', index=False)

    # одинаковые коры если есть в списках точек то чутка шевелим чтобы разные стали все
    def adjust_coordinates(self):
        epsilon = 1e-7  # очень маленькое число
        for i in range(1, len(self.points)):
            flag = 0
            while (flag == 0):
                # Проверяем совпадение координат точек
                for j in range(i):
                    flag = 1
                    if tuple(self.points[i].coords) == tuple(self.points[j].coords):
                        flag = 0
                        # Если координаты совпадают, немного изменяем их
                        for cc in range(len(self.points[i].coords)):
                            self.points[i].coords[cc] += epsilon


    # основная функция вычисляет таргет эф.                    
    def calculate_tf(self, df, p,w=1, a=1, b=-1, c=[(0,0)], q=1, r=-1, s=1, wrp=1, ret_sum=0, dop = '', nom = None):

        '''
        wrp - параметр отв. за формат вывода
        остальное какие-то бесчисленные настройки
        '''

        self.fill_boundary_points3()
        self.fill_count_clusterss2()
        self.tf = 0
        boundary_points_number = len(self.all_boundary_points)

        # Get the feature columns from the dataframe
        feature_columns_a = [col for col in df.columns if col.startswith('F1') or col.startswith('XF1')]
        feature_columns_b = [col for col in df.columns if col.startswith('F2') or col.startswith('XF2')]

        features_a = df[feature_columns_a].values
        features_b = df[feature_columns_b].values

        # Calculate the average for each feature
        feature_averages_a = np.mean(features_a, axis=0)
        feature_averages_b = np.mean(features_b, axis=0)

        # Calculate the average for each feature within each cluster
        cluster_averages = {}
        for key in self.clusters.keys():
            cluster_points_a = [features_a[self.rev_idx_map[j]] for j in self.clusters[key]]
            cluster_points_b = [features_b[self.rev_idx_map[j]] for j in self.clusters[key]]
            cluster_average_a = np.mean(cluster_points_a, axis=0)
            cluster_average_b = np.mean(cluster_points_b, axis=0)
            cluster_averages[key] = (cluster_average_a, cluster_average_b)

        lc = len([col_name1 for col_name1 in df.columns if col_name1.endswith('_CC')])
        additional_term = 1
        c_iter = 0
        if lc!=0:
            # Calculate the additional term
            additional_term = 0
            c_iter = 0
            for col_name in df.columns:
                if col_name.endswith('_CC'):
                    for key in self.clusters.keys():
                        if col_name in df.columns:
                            selected_values = df[col_name][list(self.clusters[key])]
                            sum_diff = np.abs(np.sum(selected_values) - c[c_iter+1][0])
                        additional_term += sum_diff
                    c_iter += 1
        
        if ret_sum==1:
            for key in self.clusters.keys():
                selected_values = len(self.clusters[key])
                if not (c[-1][1]) and c[-1][1] is not None:
                    sum_diff = np.abs(selected_values - c[-1][0])
                else:
                    nmean = np.mean(selected_values)
                    sum_diff = np.abs(selected_values - nmean)
                additional_term += sum_diff

        additional_term /= len(self.clusters)  # Calculate the average
        self.additional_term=additional_term ** q
        # Calculate the TF value
        u1 = 0
        u2 = 0
        for key in self.clusters.keys():
            difference_a = sum((feature_averages_a - cluster_averages[key][0]) ** 2)
            difference_b = sum((feature_averages_b - cluster_averages[key][1]) ** 2)
            self.tf += len(self.clusters[key]) * (a * sqrt(difference_a) + b * sqrt(difference_b))
            u1+=len(self.clusters[key]) * (a * sqrt(difference_a))
            u2+=len(self.clusters[key]) *  b * sqrt(difference_b)
        self.tf = self.tf ** w
        self.omega_term = self.tf
        
        self.tf *= (additional_term ** q)

        near_term = self.find_closest_point_average_distance(self.check_geom_option)

        self.tf *= (near_term ** r)
        self.tf *= (boundary_points_number ** p)

        self.near_term = near_term ** r
        self.bpn_term = boundary_points_number ** p

        farthest_term ,farthest_pairs = self.find_farthest_point_average_distance(self.check_geom_option)
        self.fath_term = farthest_term ** s
        # Умножаем TF на новый множитель
        self.tf *= (farthest_term ** s)

        if wrp:
            with open(dop+'output.txt', 'a') as file:
                if nom: file.write(nom+'\n')
                file.write(f'boundary_points_number= {boundary_points_number}\n')
                file.write(f'bpn^{p} = {boundary_points_number**(p)}\n')

                file.write(f'mult.term **{q} = { (additional_term ** q)}\n')
                file.write(f'mean_near **{r} = {(near_term ** r)}\n')
                file.write(f'farthest_term **{s} = {farthest_term ** s}\n')
                #file.write(f'pre_tf+ = {u1}\n')
                #file.write(f'pre_tf- = {difference_b}\n')
                file.write(f'self.tf = {self.tf}\n')
                file.write(f'__\n')
            
            with open('farthest_pairs.txt', 'w') as f:
                for pair in farthest_pairs:
                    f.write(f"Точка {pair[0]} - Точка {pair[1]}: Расстояние {pair[2]}\n")

        else:
            with open('output1.txt', 'a') as file:
                file.write(f'boundary_points_number= {boundary_points_number}\n')
                file.write(f'bpn^{p} = {boundary_points_number**(p)}\n')

                file.write(f'mult.term **{q} = { (additional_term ** q)}\n')
                file.write(f'mean_near **{r} = {(near_term ** r)}\n')
                file.write(f'farthest_term **{s} = {farthest_term ** s}\n')
                file.write(f'pre_tf+ = {u1}\n')
                file.write(f'pre_tf- = {difference_b}\n')
                file.write(f'self.tf = {self.tf}\n')
                file.write(f'__\n')

        if self.po:
            # Create a DataFrame with the data
            fdata = {

                'Farthest_Pair': farthest_pairs
            }
            fdf = pd.DataFrame(fdata)

            # Save the DataFrame to an Excel file
            fdf.to_excel('Farthest_output.xlsx', index=False)

        return self.tf


    #установочная перед калькуляциями
    def setets(self, excel_file=None):
        self.adjust_coordinates()
        if not self.g_option:
            self.triangulation, _, self.edge_indexs = build_delaunay_triangulation(self.points)
            self.all_boundary_points = find_boundary_points(self.edge_indexs, self.points, self.rev_idx_map)
            self.edgemap = {}
            for point in self.points:
                key = point.index[0]
                values = [pair[0] if pair[1] == key else pair[1] for pair in self.edge_indexs if key in pair]
                self.edgemap[key] = values

            # для тестов
            '''
            with open('test_outp.txt', 'a') as file:
                file.write('edgemap: ')
                file.write(str(self.edgemap) + '\n')
                file.write('edgeindexs: ')
                file.write(str(self.edge_indexs) + '\n')
                file.write('all_boundary_points_ids: ')
                file.write(str([p.index[0] for p in self.all_boundary_points]) + '\n')
            '''
            self.save_graph()
        else:
            # Вызываем метод build_shortest_paths для построения кратчайших путей
            graph_built=False
            attempts=0
            excel_file_old=excel_file
            while not graph_built:
                attempts+=1
                excel_file_new = "{0}_{2}{1}".format(*os.path.splitext(excel_file) + (str(attempts),))
                shutil.copyfile(excel_file_old, excel_file_new)
                graph_built = self.build_weighted_graph(excel_file_new)
                if not graph_built:
                    print('Graph was corrected. Check', excel_file_new, 'for changes. It can be corrected manually.')
                    input(f'Press any key when {excel_file_new} ready to process')
                if excel_file_old != excel_file: os.remove(excel_file_old)
                excel_file_old=excel_file_new

            predecessors, matrix = self.build_shortest_paths()
            print(matrix)

            nearest_path = input("создать ли файл оптимальных маршрутов? (Y/N) ").upper() == 'Y'
            if nearest_path:
                nearest_path_file = input("Путь к файлу для сохранения (по умолчанию shortest_paths.xlsx) ")
                if '.xlsx' not in nearest_path_file:
                    nearest_path_file = 'shortest_paths.xlsx'
                shortest_paths=[]
                for p1, p2 in  itertools.combinations(self.points, 2):
                    p1=p1.index[0]
                    p2=p2.index[0]
                    spath = ','.join([str(int(v)) for v in nx.reconstruct_path(p1, p2, predecessors)])
                    shortest_paths.append([p1, p2, spath, matrix[p1][p2]])
                pd.DataFrame(shortest_paths, columns=['id_нач.тчк1', 'id_нач.тчк2', 'path',  'weight']).to_excel(nearest_path_file)
                input(f'Press any key when {nearest_path_file} ready to process')
                shortest_paths = pd.read_excel(nearest_path_file)

            # Заполняем матрицу расстояний self.table из матрицы matrix
            for i, p1 in enumerate(self.wGraph.nodes):
                for j, p2 in enumerate(self.wGraph.nodes):
                    if not nearest_path:
                        self.table[self.rev_idx_map[p1], self.rev_idx_map[p2]] = matrix[p1][p2]
                    else:
                        self.table[self.rev_idx_map[p1], self.rev_idx_map[p2]] = shortest_paths.loc[(((shortest_paths['id_нач.тчк1']==p1) & (shortest_paths['id_нач.тчк2']==p2))|((shortest_paths['id_нач.тчк1']==p2) & (shortest_paths['id_нач.тчк2']==p1))), 'weight'].values[0] if p1!=p2 else 0

    def calculate_table(self):
        self.table = [[self.dist(self.points[i], self.points[j]) for j in range(i+1, len(self.points))] for i in range(len(self.points))]

    #расстояние между парой точек. если по графу (self.g_option = true) - то применяется кратчайший по графу. если нет - просто дистанция евклидова
    def dist(self, p1, p2):
        """
        Calculates the distance between 2 points in R^n
        :param p1: Point 1
        :param p2: Point 2
        :return: distance
        """
        if not self.g_option:
            l = [c1 - c2 for c1, c2 in zip(p1.coords, p2.coords)]
            return sqrt(sum(map(lambda x: x ** 2, l)))
        else:
            # Получаем индексы точек p1 и p2
            idx_p1 = self.rev_idx_map[p1.index[0]]
            idx_p2 = self.rev_idx_map[p2.index[0]]
            
            # Возвращаем расстояние из self.table
            return self.table[idx_p1, idx_p2]

    # 2, первой и нет
    def fill_count_clusterss2(self):
        self.clusters={}
        c=set()
        t=-1
        u=0
        for p in self.points:
            t = p.index[1]
            c.add(t)
            if t in self.clusters.keys():
                self.clusters[t].add(p.index[0])
            else:
                self.clusters[t]=set([p.index[0]])
                u+=1
        self.maxclustid=max(c)
        self.clastnum = u
    
    def ds_proc(self,df, broken_rows):
        ds = df
        br = [i-2 for i in broken_rows]
        ds.drop(ds.index[br], inplace=True)
        return ds
    
    # первых двух тоже нет))
    def fill_boundary_points3(self):
        self.all_boundary_points = find_boundary_points(self.edge_indexs, self.points, self.rev_idx_map)
    

    def find_part_boundary_points_for_id(self, point_id):
        pt = self.points[self.rev_idx_map[point_id]]
        clust_num = pt.index[1]
        edges = self.edgemap[point_id]
        edges.append(point_id)
        boundary_points = []
        non_boundary_points = []
        for edge in edges:
            p_e = self.points[self.rev_idx_map[edge]]
            if p_e.index[1] != clust_num:
                boundary_points.append(p_e)
            else:
                cc = set()
                cm = 0
                edges1 = self.edgemap[edge]
                for e1 in edges1:
                    p_e1 = self.points[self.rev_idx_map[e1]]
                    edge_clust = p_e1.index[1]
                    if edge_clust not in cc:
                        cc.add(edge_clust)
                        if len(cc)>1:
                            boundary_points.append(p_e)
                            cm = 1
                            break
                if cm == 0:
                    non_boundary_points.append(p_e)
        
        for p in boundary_points:
            self.all_boundary_points.add(p)
        for p in non_boundary_points:
            self.all_boundary_points.discard(p)
    
    def rand_sel_boundary(self):
        #return random boundary point id
        self.fill_boundary_points3()
        if len(self.all_boundary_points)==0:
            print("Остался один кластер.Программа завершается...")    
            time.sleep(2)
            exit()
        b = random.choice(list(self.all_boundary_points))
        return b

    # вспомогательные для tf

    def check_geom_condition(self,pa,pb, pb_bounds):
        bpa,_ = find_boundary_points_for_id(pa.index[0], self.points, self.rev_idx_map, self.edgemap)
        bpb = pb_bounds.union(bpa)
        c=1
        dst = self.dist(pa,pb)
        if len([p for p in bpb if (p != pa and p !=pb and  self.dist(p,pa)+self.dist(p,pb) < dst)]) > 0:
            c=0
        return c

    def check_geom_condition_f(self,pa,pb, some_points):
        c=1
        dst = self.dist(pa,pb)
        if len([p for p in some_points if (p != pa and p !=pb and  self.dist(p,pa)+self.dist(p,pb) < dst)]) > 0:
            c=0
        return c

    def find_closest_point_average_distance(self, check_geom_option=False):
        all_distances = []
        for point in self.points:
            distances_to_boundary, _ = find_boundary_points_for_id(point.index[0], self.points, self.rev_idx_map, self.edgemap)
            # Рассчитываем расстояния до всех граничных точек и находим минимальное
            if len(distances_to_boundary)>0:
                #min_distance = min(self.dist(point, boundary_point) for boundary_point in distances_to_boundary)
                min_distance, bpoint = min(((self.dist(point, boundary_point), boundary_point) for i, boundary_point in enumerate(distances_to_boundary)), key=lambda x: x[0])
                if check_geom_option:
                    if self.check_geom_condition(bpoint, point, distances_to_boundary):
                    # Сохраняем минимальное расстояние для данной точки
                        all_distances.append(min_distance)
                else:
                    all_distances.append(min_distance)

        # Возвращаем среднее расстояние
        #average_distance = sum(all_distances) / len(all_distances) if len(all_distances)>0 else 1
        # Среднеквадратичное
        average_distance = np.sqrt(sum(x**2 for x in all_distances) / len(all_distances)) if len(all_distances) > 0 else 1
        return average_distance

    def find_farthest_point_average_distance(self, check_geom_option=False):
        all_distances = []
        farthest_pairs = []  # Список для сохранения точек с максимальным расстоянием

        for point in self.points:
            internal_points,internal_points_ids, all_b_points = find_internal_points_for_id(point.index[0], self.points, self.rev_idx_map, self.edgemap)
            
            # Рассчитываем расстояния до всех отфильтрованных точек и находим максимальное
            max_distance = None
            if internal_points:
                '''
                maxx_distance, farthest_point = max(
                    ((self.dist(point, internal_point), internal_point) for internal_point in internal_points),
                    key=lambda x: x[0],
                    default=(None, None)
                )

                if maxx_distance is not None:
                    if self.check_geom_condition:
                        if self.check_geom_condition_f(farthest_point, point, all_b_points):
                            all_distances.append(maxx_distance)
                            farthest_pairs.append((point.index[0], farthest_point.index[0],maxx_distance))
                '''
            
            
                # Сортируем internal_points по расстоянию до point в порядке убывания
                sorted_distances_points = sorted(
                    ((self.dist(point, internal_point), internal_point) for internal_point in internal_points),
                    key=lambda x: x[0],
                    reverse=True
                )

                # Проходим по отсортированному списку
                for mdistance, potential_farthest_point in sorted_distances_points:
                    if mdistance is not None:
                        if self.check_geom_condition:
                            if self.check_geom_condition_f(potential_farthest_point, point, all_b_points):
                                all_distances.append(mdistance)
                                farthest_pairs.append((point.index[0], potential_farthest_point.index[0], mdistance))
                                break
        
        # Рассчитываем среднеквадратическое значение расстояний
        if all_distances:
            aver_farthest_dist_internal = (sum(d**2 for d in all_distances) / len(all_distances))**0.5
        else:
            aver_farthest_dist_internal = 0.0000000000000000000001
        
        return aver_farthest_dist_internal,farthest_pairs

    # Регион экшн функций - отдельная трансляция, серии, все дела

    def translation(self,point_id, new_clust_id):
        cur_point = self.points[self.rev_idx_map[point_id]]
        old_clust_id = cur_point.index[1]
        cur_point.index[1] = new_clust_id
        self.clusters[old_clust_id].remove(point_id)

        if len(self.clusters[old_clust_id])==0:
            del self.clusters[old_clust_id]
            if len(self.clusters) == 1:
                print("Остался один кластер. Программа завершается. Подождите...")
                time.sleep(2)
                exit()
        if new_clust_id in self.clusters.keys():
            self.clusters[new_clust_id].add(point_id)
        else: 
            self.clusters[new_clust_id]=set([point_id])

        self.find_part_boundary_points_for_id(point_id)

    def simple_transition(self,alpha, wrp):
        curr_point = self.rand_sel_boundary()
        point_id = curr_point.index[0]
        is_new=False
        rn = random.random()

        if rn>alpha:
            bp_ids = list(self.edgemap[point_id])
            bp = [self.points[self.rev_idx_map[idz]] for idz in bp_ids if self.points[self.rev_idx_map[idz]].index[1] != curr_point.index[1]]
            distances = [self.dist(self.points[self.rev_idx_map[point_id]], boundary_point) for boundary_point in bp]
            # Находим индекс точки с минимальным расстояниемis_new
            min_dist_index = distances.index(min(distances))
            # Берем точку с минимальным расстоянием
            boundary_point = bp[min_dist_index]
            new_clust_num = boundary_point.index[1]
        else:
            is_new=True
            self.maxclustid+=1
            new_clust_num = self.maxclustid

        self.translation(point_id,new_clust_num)
        if wrp:
            with open('output.txt', 'a') as file:
                file.write(f'SIMPLE\npoint to trans id {point_id}  \nnew clust num: {new_clust_num} \nis_new? {is_new} \n')
        else:
            with open('output1.txt', 'a') as file:
                file.write(f'SIMPLE\npoint to trans id {point_id}  \nnew clust num: {new_clust_num} \nis_new? {is_new} \n')
        return point_id, new_clust_num, curr_point

    def correcting_transition(self,alpha,wrp):
        curr_point = self.rand_sel_boundary()
        point_id = curr_point.index[0]
        is_new=False
        new_clust_num=-1
        bp_ids = list(self.edgemap[point_id])
        bp = [self.points[self.rev_idx_map[idz]] for idz in bp_ids if self.points[self.rev_idx_map[idz]].index[1] != curr_point.index[1]]

        distances = [self.dist(curr_point, boundary_point) for boundary_point in bp]
        # Находим индекс точки с минимальным расстоянием
        min_dist_index = distances.index(min(distances))
        # Берем точку с минимальным расстоянием
        boundary_point = bp[min_dist_index]
        point_to_trans_id = boundary_point.index[0]

        rn = random.random()
        if rn>alpha:
            new_clust_num = curr_point.index[1]
        else:
            is_new=True
            self.maxclustid+=1
            new_clust_num = self.maxclustid
        if wrp:
            with open('output.txt', 'a') as file:
                file.write(f'CORRECTING\npoint to trans id {point_to_trans_id} \nnew clust num: {new_clust_num}\nis_new? {is_new} \n')
        else:
            with open('output1.txt', 'a') as file:
                file.write(f'CORRECTING\npoint to trans id {point_to_trans_id} \nnew clust num: {new_clust_num}\nis_new? {is_new} \n')
        
        self.translation(point_to_trans_id,new_clust_num)
        return point_to_trans_id, new_clust_num, boundary_point
    
    def start_series(self,alpha,tp,p_tf,wrp=1): 
        self.fill_boundary_points3()
        self.fill_count_clusterss2()        
        r1=random.random()
        pt_id=-1
        if r1<tp:
            pt_id, ncn, pt=self.correcting_transition(alpha,wrp) 
        else:
            pt_id,ncn, pt=self.simple_transition(alpha,wrp)
        
        self.fill_boundary_points3()
        self.fill_count_clusterss2() 

        u=0
        if (pt in self.all_boundary_points):
            u=1
            

        return u, pt_id, ncn, pt
            
    def simple_series(self, length, alpha,sn,tp, p_tf,o_file,wrp,fc):
        if_need, pt_id, new_clust_num, pt = self.start_series(alpha,tp,p_tf,wrp)
        c12=0
        point_id = pt_id
        cur_ser_points = set([pt])
        ncn = new_clust_num
        list_of_all_used_points = [pt]
        with open('output_ns.txt', 'a') as file:
            file.write(f"Start Series: if_need = {if_need} , pt_id = {pt_id} \n, new_clust_num = {new_clust_num} \n")

        if if_need:

            while(c12<length and len(cur_ser_points)>0):
                c12+=1
                rnd_point = random.choice(list(cur_ser_points))
                rnd_point_id = rnd_point.index[0]
                bp_ids = list(self.edgemap[rnd_point_id])
                bp = [self.points[self.rev_idx_map[idz]] for idz in bp_ids]
                bp = [kbp for kbp in bp if kbp.index[1] != new_clust_num]
                point_to_trans = random.choice(bp)
                point_to_trans_id = point_to_trans.index[0]
                
                self.translation(point_to_trans_id,ncn)
                self.fill_boundary_points3()

                cur_ser_points.add(point_to_trans)
                list_of_all_used_points.append(point_to_trans)

                ff = set([point for point in list_of_all_used_points if point in self.all_boundary_points])
                cur_ser_points=ff
        self.fill_boundary_points3()
        self.find_clusters()
        self.fill_count_clusterss2()

        with open(o_file, 'a') as file:
            file.write(f'trans_points_nums: {[ p.index[0] for p in cur_ser_points]}\n')        
        with open(o_file, 'a') as file:
            file.write(f'real ser. length: {c12+1}\n')
        with open(o_file, 'a') as file:
            file.write(f'clust_trans: {ncn}\n')
        with open('output_ns.txt', 'a') as file:
            file.write(f"real ser. length: {c12+1}, all_used_points: {[pnt.index[0] for pnt in list_of_all_used_points]}\n")

    
    #................... GRAPH UTILS ..................... #

    def save_graph(self):
        graph = pd.DataFrame([[point, key, self.dist(self.points[self.rev_idx_map[point]], self.points[self.rev_idx_map[key]])] for key in self.edgemap.keys() for point in self.edgemap[key] ],
                             columns=["Id_Point1", "Id_Point2", "edge"])
        graph.to_excel('graph.xlsx')

    def build_graph(self):
        # Создаем пустой граф в виде словаря, где ключи - вершины, а значения - соседи
        graph = {point.index[0]: [] for point in self.points}
        
        # Добавляем ребра (edges) на основе self.edgemap
        for point in self.points:
            ind = point.index[0]
            for neighbor in self.edgemap[ind]:
                # Проверяем, принадлежат ли точки одному кластеру
                if self.points[self.rev_idx_map[ind]].index[1] == self.points[self.rev_idx_map[neighbor]].index[1]:
                    graph[ind].append(neighbor)
                    graph[neighbor].append(ind)  # добавляем обратное ребро, так как граф ненаправленный
        return graph

    def save_shortest_paths(nearest_path_file):
        graph = pd.DataFrame(
            [[point, key, self.dist(self.points[self.rev_idx_map[point]], self.points[self.rev_idx_map[key]])] for key
             in self.edgemap.keys() for point in self.edgemap[key]],
            columns=["Id_Point1", "Id_Point2", "edge"])
        graph.to_excel('graph.xlsx')
  
    def build_weighted_graph(self, excel_file):
        success=True
        points=[p.index[0] for p in self.points]

        # Считываем данные из Excel-файла
        df = pd.read_excel(excel_file)
        
        # Создаем пустой неориентированный граф
        G = nx.Graph()
        edge_indexs_set = set()

        for point in self.points:
            G.add_node(point.index[0])
        
        # Заполняем граф на основе данных из Excel
        for _, row in df.iterrows():
            id_point1, id_point2, edge = row["Id_Point1"], row["Id_Point2"], row["edge"]
            if id_point1 not in points or id_point2 not in points:
                print('Точка', id_point1, 'or', id_point2, 'отсутствует в исходных данных')
                continue

            edge_indexs_set, G = self.add_edge(edge_indexs_set, G, id_point1, id_point2, edge)

        #check if all points in graph
        missed_points = [p.index[0] for p in points if p not in df["Id_Point1"].values and p not in df["Id_Point2"].values]
        if len(missed_points)>0:
            for p in missed_points:
                print('Точка', p, 'отсутствует в графе')

        #Проверяем связанность графа
        connected_components=list(nx.connected_components(G))
        if len(connected_components)>1:
            success=False
            print(f'Граф не связный,  был автоматически сформирован вариант  приведения к единой компоненте связности и сохранен в другой файл {excel_file}. просьба  проверить  файл изменений, принять изменения или внести коррективы')
            max_edge=max([edge[2] for edge in G.edges.data("weight", default=0)])
            df['origin']=['initial']*df.shape[0]
            for i in range(len(connected_components)-1):
                df.loc[df.shape[0]]=[list(connected_components[i])[0],
                                                   list(connected_components[i+1])[0],
                                                   max_edge*50, 'automatically added']
                #edge_indexs_set, G = self.add_edge(edge_indexs_set, G,
                #                                   list(connected_components[i])[0],
                #                                   list(connected_components[i+1])[0],
                #                                   max_edge*50)
            df.to_excel(excel_file)
        else:
            # Сохраняем созданный граф в атрибуте self.wGraph
            self.edge_indexs = edge_indexs_set
            self.wGraph = G
        return success

    def add_edge(self, edge_indexs_set, G, id_point1, id_point2, edge):
        G.add_edge(id_point1, id_point2, weight=edge)

        edge_indexs_set.add((id_point1, id_point2))
        edge_indexs_set.add((id_point2, id_point1))

        # Обновляем self.edgemap для точки id_point2
        if id_point1 in self.edgemap:
            self.edgemap[id_point1].append(id_point2)
        else:
            self.edgemap[id_point1] = [id_point2]

        # Обновляем self.edgemap для точки id_point2
        if id_point2 in self.edgemap:
            self.edgemap[id_point2].append(id_point1)
        else:
            self.edgemap[id_point2] = [id_point1]

        return edge_indexs_set, G

    def build_shortest_paths(self):
        if self.wGraph is None:
            raise ValueError("Weighted graph is not built. Please call build_weighted_graph first.")
        
        # Находим кратчайшие пути между всеми парами вершин с помощью алгоритма Флойда-Уоршалла
        #matrix = nx.floyd_warshall_numpy(self.wGraph)
        predecessors, matrix = nx.floyd_warshall_predecessor_and_distance(self.wGraph)

        return predecessors, matrix
   
    def find_connected_components(self, graph):
        # Находим компоненты связности в графе
        visited = set()
        components = []
        
        for vertex in graph:
            if vertex not in visited:
                component = self.dfs(graph, vertex, visited)
                components.append(component)
        return components
    
    def dfs(self, graph, start_vertex, visited):
        # Рекурсивный обход в глубину для поиска компоненты связности
        stack = [start_vertex]
        component = []
        
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                component.append(vertex)
                stack.extend(graph[vertex])
        return component
    
    def find_clusters(self):
        # Построение графа
        graph = self.build_graph()
        
        # Находим компоненты связности
        components = self.find_connected_components(graph)
        
        # Создаем кластера на основе компонент связности
        self.clusters = {i: set(component) for i, component in enumerate(components)}

        for cluster in self.clusters:
            for cluster_id in self.clusters[cluster]:
                self.points[self.rev_idx_map[cluster_id]].index[1] = cluster

    def set_new_clusters(self, new_clusters):
        self.clusters = new_clusters
        for p in self.points:
            p.index[1] = [key for key in new_clusters.keys() if p.index[0] in new_clusters[key]][0]
        self.fill_count_clusterss2()
        self.fill_boundary_points3()

# _________________________END OF CLASS TABLE_________________________


def prepare():
    pd.options.mode.chained_assignment = None #забыл что это
    filename = input('Enter the file name: ')

    try:
        df, src, broken_rows, i_s, ret_sum, c_values1 = process_file(filename) #src это точки if_start наличие колонки граничных

    except AttributeError:
        print('Incorrect input file')
        return
    t = Table(src)

    ds = t.ds_proc(df,broken_rows)
    output_file = 'output.txt'

    if os.path.exists(output_file):
        os.remove(output_file)
    
    output_file = 'o_log.xlsx' 

    if os.path.exists(output_file):
        os.remove(output_file)

    output_file = 'output_ds.xlsx' 

    if os.path.exists(output_file):
        os.remove(output_file)

    ds=df

    for i in range(len(ds)):
        ds.at[i, 'ID'] = t.points[i].index[0]

    fc=connected_components_autosplit_ui()

    parameters = get_parameters()
    print(parameters)
    alpha = float(parameters['alpha'])
    tp = float(parameters['tp'])
    ser = int(parameters['ser'])
    maxlen = int(parameters['maxlen'])
    p_tf = float(parameters['p_tf'])
    s1 = parameters['s1']
    a = float(parameters['a'])
    b = float(parameters['b'])
    c = c_values1
    q = float(parameters['q'])
    r = float(parameters['r'])
    s = float(parameters['s'])
    a_th = float(parameters['a_th'])
    o_th = float(parameters['o_th'])
    b_th = float(parameters['b_th'])
    n_th = float(parameters['n_th'])
    f_th = float(parameters['f_th'])

    n_lust = int(parameters['n_lust'])
    thresh_bond = float(parameters['r_threshold'])
    max_ser_bond = float(parameters['up_lim'])
    w = float(parameters['omega'])
    folder_path='kartinki'
    shutil.rmtree(folder_path, ignore_errors=True)
    os.makedirs(folder_path, exist_ok=True)

    return t, ds, alpha, tp,ser,maxlen,p_tf,s1,a,b,c,q,r,s,n_lust,thresh_bond,max_ser_bond, folder_path,ret_sum,filename,w,fc,a_th,o_th,b_th,n_th,f_th

def postfix(t, filename,s1,folder_path,ds, prefix=''):
    t.fill_count_clusterss2()
    #t.setets()
    t.fill_boundary_points3()
    t.po=1

    input_file = 'output.txt'
    output_file = 'o_log.xlsx'
    if prefix!='merged_': process_data(input_file, output_file)
    if len(t.points[0].coords)==2:
        visualize_clusters2d(t,folder_path,prefix+'end')

    t.process_dataframe(ds,filename)


# _______________________MAIN_______________________#


def process_file(path):
    df=''
    if_start = 1
    if path.endswith('.xlsx'):
        for i in range(1, 5):
            try:
                df = pd.read_excel(path) 
            except ValueError:
                continue
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        print('Unknown filetype, exiting')
        exit(1)
    id_index = 0
    cluster_id_index = 0
    id_coords = np.array([], dtype=int)

    def get_row_item(row, index):
        return row[1][df.columns[index]]
    
    assign_column_roles_ui(df)
    ret_sum, c_values = assign_C_roles_ui(df)

    for ind, elem in enumerate(df.columns):
        if elem == 'ID':
            id_index = ind
        elif elem == 'Clust_ID':
            cluster_id_index = ind        
        elif elem.startswith('Is_boundary'):
            if_start = 0
        else:
            if elem.startswith('X'):
                id_coords = np.append(id_coords, ind)


    points = []
    ids = np.array([], dtype=int)
    affected_rows = np.array([], dtype=int)
    c=0
    for i, row in enumerate(df.iterrows()):
        c=0
        coords = np.zeros(id_coords.size)
        correct = True
        if get_row_item(row, id_index) in ids:
            affected_rows = np.append(affected_rows, i + 2)
            continue
        for ind in range(len(row[1])):
            elem = get_row_item(row, ind)
            if not check_digit(str(elem)):
                affected_rows = np.append(affected_rows, i + 2)
                correct = False
                break
            if ind in id_coords:
                coords[c] = elem
                c+=1
        if correct:
            ids = np.append(ids, get_row_item(row, id_index))
            points.append(Point(int(get_row_item(row, id_index)), int(get_row_item(row, cluster_id_index)), coords))
    if len(affected_rows) > 0:
        print('Incorrect rows are: ')
        print(*affected_rows)
        ans = input('If you want to continue, ignoring them, type "yes"')
        if ans.lower() != 'yes':
            raise AttributeError

    ds = df   
   
    return ds, points, affected_rows, if_start, ret_sum, c_values

def main():
    t, ds, alpha, tp,ser,maxlen,p_tf,s1,a,b,c,q,r,s,n_lust,thresh_bond,max_ser_bond,folder_path,ret_sum,filename, w,fc, a_th,o_th,b_th,n_th,f_th =prepare()
    
    base_len = maxlen
    base_alpha = alpha
    t.fill_count_clusterss2()
    tg_option, int_search_option, check_geom_option = get_user_options()
    t.g_option = tg_option
    t.check_geom_option = check_geom_option
    if t.g_option:
        graph_file = 'graph.xlsx'
        if not t.g_option: t.setets(graph_file)
        graph_file_input = input("Введите путь к файлу Excel для для построения графа (по умолчанию используется graph.xlsx с графом Воронова): ")
        if os.path.exists(graph_file_input):
            graph_file = graph_file_input
        t.g_option = True
        t.setets(graph_file)
    else:
        t.setets()
        
    t.fill_boundary_points3()
    t.calculate_tf(ds,p_tf,w, a, b, c, q,r,s,1, ret_sum)
    if len(t.points[0].coords)==2:
        visualize_clusters2d(t, folder_path, 'start')

    maxt = {
        'at_max': -float('inf'),
        'ot_max': -float('inf'),
        'neart_max': -float('inf'),
        'bpnt_max': -float('inf'),
        'fatht_max': -float('inf') 
        }
    
    
    # просто запускает серию с параметрами указанными
    def splitius(l_len,alpha,tf_arr,t,fc, maxt):  #локальная серия
        #l_len1 =1 + np.random.binomial(l_len-1, 0.5)
        interval_size = max(1, (l_len-1) // num_of_intervals)
        intervals = [(i*interval_size, (i+1)*interval_size) for i in range(num_of_intervals)]
        # Randomly select one element from each interval
        selected_elements = [np.random.randint(low=interval[0], high=interval[1]+1) for interval in intervals]
        l_len1 = sum(selected_elements)+1

        old_tf = t.calculate_tf(ds,p_tf,w, a, b, c, q,r,s,0,ret_sum)
        temp_t3 = copy.deepcopy(t)
        wrpt=0

        t.simple_series(l_len1,alpha,i,tp,p_tf,'output1.txt',0,fc)

        new_tf = t.calculate_tf(ds,p_tf,w, a, b, c, q,r,s,0, ret_sum)

        with open('output1.txt', 'a') as file:
            file.write(f'alpha,l_len: {alpha} {l_len}\n P New_TF= {new_tf} \nOld_TF= {old_tf}\n ')

        if new_tf <= old_tf:  # If the new result is not greater than the original result
            if (t.check_tf_is_better(maxt['at_max'], maxt['ot_max'], maxt['bpnt_max'], maxt['neart_max'], maxt['fatht_max'], a_th, o_th,b_th,n_th,f_th)):
                t = temp_t3

        update_max_values(maxt, t)

        #________params_recalc___________
        tf_arr.append(max(new_tf,old_tf))
        return tf_arr[-1],t,tf_arr

    tfs=[] #храним tf-значения по итерациям
    tfs.append(t.calculate_tf(ds,p_tf,w, a, b, c, q,r,s,1,ret_sum))
    c_start=0
    #____________________main loop_____________________
    lust_count=3
    i=0

    if not int_search_option:
        n_lust = 1
    if ser == 0:
        i=20
    #print(n_lust)
    while i<ser+2*n_lust:
        i+=1
        print('ser. num = ', i)

        #l_len =1+ np.random.binomial(maxlen-1, 0.5)
        interval_size = max(1, (maxlen-1) // num_of_intervals)
        # Randomly select one element from each interval
        selected_elements = [np.random.randint(0, interval_size+1) for i in range(num_of_intervals)]
        l_len = sum(selected_elements)+1
        with open('output.txt', 'a') as file:
            file.write(f'\nNUM_SER {i} \nCURRENT_MAX_LEN {l_len+1} \n')

        with open('output_ns.txt', 'a') as file:
            file.write(f'\nNUM_SER {i} \nCURRENT_MAX_LEN {l_len+1} \n')
            file.write(f'SELECTED_ELEMENTS: {selected_elements}\n')
            
        old_tf = t.calculate_tf(ds,p_tf,w, a, b, c, q,r,s,1,ret_sum)
        temp_t = copy.deepcopy(t)
        t.simple_series(l_len,alpha,i,tp,p_tf, 'output.txt',1,fc)

        new_tf = t.calculate_tf(ds,p_tf,w, a, b, c, q,r,s,1, ret_sum)

        with open('output.txt', 'a') as file:
            file.write(f'New_TF= {new_tf} \nOld_TF= {old_tf}\n')
        if new_tf < old_tf:  # If the new result is not greater than the original result  
                with open('output.txt', 'a') as file:
                    file.write('CANCELED\n')
                    t = copy.deepcopy(temp_t)
        else:
            if t.check_tf_is_better(maxt['at_max'], maxt['ot_max'], maxt['bpnt_max'], maxt['neart_max'], maxt['fatht_max'], a_th, o_th,b_th,n_th,f_th):
                with open('output.txt', 'a') as file:
                    file.write('ACCEPTED\n')
            else:
                with open('output.txt', 'a') as file:
                    file.write('CANCELED\n')
                    t = copy.deepcopy(temp_t)
        with open('output.txt', 'a') as file:    
            file.write(f'!!!!\n')
        
        update_max_values(maxt, t)
        

        
        if (i%kartinka_num) == kartinka_num-1:
            nom=f'{i}'
            if len(t.points[0].coords)==2:
                visualize_clusters2d(t, folder_path, nom)
                t.calculate_tf(ds, p_tf, w, a, b, c, q, r, s, 1, ret_sum, dop='kartinki/', nom=nom)

        #________params_recalc___________

        tfs.append(max(new_tf,old_tf))
        if int_search_option:
            if i>n_lust+1:
                abs_delta_tf = (tfs[-1]-tfs[0]) / i
                current_delta_tf = (tfs[-1]-tfs[i-n_lust-1])/(n_lust)
                with open('output.txt', 'a') as file:
                    file.write(f'abs_delta_tf! {abs_delta_tf}\n')
                    file.write(f'curr_delta_tf! {current_delta_tf}\n')
            if i<(lust_count+n_lust):
                continue
            if current_delta_tf < abs_delta_tf*thresh_bond:
                with open('output.txt', 'a') as file:
                    file.write(f'slow growth rate detected! iter = {i}, alpha= {alpha}, ser_len = {maxlen}\n')
                lust_count=i+n_lust

                alphas = [alpha, low_param*alpha]
                if high_param*alpha < 1:
                    alphas.append(high_param*alpha)
                ser_lengths = [maxlen]
                if (high_param*maxlen) < max_ser_bond:
                    ser_lengths.append(int(high_param*maxlen))
                ser_lengths.append(int(low_param*maxlen))
                max_tf=-100000
                t_it = 0
                tfs_it = []
                tt = 0
                dds = 0
                alpha0=0
                print("slow start. tf = ", t.tf)
                mem = tfs[-1]
                for alpha1 in alphas:
                    for sl in ser_lengths:
                        tf_arr = []
                        t2 = copy.deepcopy(t)
                        maxtt = copy.deepcopy(maxt)
                        for j in range(n_lust):
                            dds,t2, tf_arr = splitius(sl,alpha1,tf_arr,t2,fc,maxtt)
                        with open('output1.txt', 'a') as file:
                            file.write(f'alt1 {alpha1} tf {t2.tf} {dds} len_arr {len(tf_arr)} {tf_arr}\n')

                        if dds > max_tf:
                            
                            t_it = copy.deepcopy(t2)
                            maxlen = max(1,sl)
                            tfs_it = copy.deepcopy(tf_arr)
                            alpha0 = alpha1
                            max_tf = dds
                t = t_it
                i+=n_lust
                alpha = alpha0
                for l1 in tfs_it:
                    tfs.append(l1)
                print("slow end. tf = ", t.tf)
                print(f'new params selected! iter = {i}, alpha= {alpha}, ser_len = {maxlen}\n')
                if abs(mem - tfs[-1]) < 0.0000001:
                    alpha = base_alpha
                    maxlen = base_len
                    print(f'Поиск был неуспешен, восстановлены базовые параметры, alpha= {base_alpha}, ser_len = {base_len}\n')
                with open('output.txt', 'a') as file:
                    file.write(f'new params selected! iter = {i}, alpha= {alpha}, ser_len = {maxlen}\n')

#Дополнительные серии

    print(f'Total clusters: {t.maxclustid + 1}\n')
    add_ser1 = int(input('Введите дополнительное количество серий\n'))
    maxlen = int(input('Введите максимальную длину дополнительных серий\n'))

    print(maxlen)
    while i < ser + 2 * n_lust + add_ser1:
        i += 1
        print('ser. num = ', i)
        # l_len =1+ np.random.binomial(maxlen-1, 0.5)
        interval_size = max(1, (maxlen - 1) // num_of_intervals)
        # Randomly select one element from each interval
        selected_elements = [np.random.randint(0, interval_size + 1) for i in range(num_of_intervals)]
        l_len = sum(selected_elements) + 1
        with open('output.txt', 'a') as file:
            file.write(f'\nNUM_SER {i} \nCURRENT_MAX_LEN {l_len + 1} \n')

        with open('output_ns.txt', 'a') as file:
            file.write(f'\nNUM_SER {i} \nCURRENT_MAX_LEN {l_len + 1} \n')
            file.write(f'SELECTED_ELEMENTS: {selected_elements}\n')

        old_tf = t.calculate_tf(ds, p_tf, w, a, b, c, q, r, s, 1, ret_sum)
        temp_t = copy.deepcopy(t)
        t.simple_series(l_len, alpha, i, tp, p_tf, 'output.txt', 1, fc)

        new_tf = t.calculate_tf(ds, p_tf, w, a, b, c, q, r, s, 1, ret_sum)

        with open('output.txt', 'a') as file:
            file.write(f'New_TF= {new_tf} \nOld_TF= {old_tf}\n')
        if new_tf < old_tf:  # If the new result is not greater than the original result
            with open('output.txt', 'a') as file:
                file.write('CANCELED\n')
                t = copy.deepcopy(temp_t)
        else:
            if t.check_tf_is_better(maxt['at_max'], maxt['ot_max'], maxt['bpnt_max'], maxt['neart_max'],
                                    maxt['fatht_max'], a_th, o_th, b_th, n_th, f_th):
                with open('output.txt', 'a') as file:
                    file.write('ACCEPTED\n')
            else:
                with open('output.txt', 'a') as file:
                    file.write('CANCELED\n')
                    t = copy.deepcopy(temp_t)
        with open('output.txt', 'a') as file:
            file.write(f'!!!!\n')

        update_max_values(maxt, t)

        if (i % kartinka_num) == kartinka_num - 1:
            nom = f'{i}'
            if len(t.points[0].coords) == 2:
                visualize_clusters2d(t, folder_path, nom)
                t.calculate_tf(ds, p_tf, w, a, b, c, q, r, s, 1, ret_sum, dop='kartinki/', nom=nom)

        # ________params_recalc___________

        tfs.append(max(new_tf, old_tf))
        if int_search_option:
            if i > n_lust + 1:
                abs_delta_tf = (tfs[-1] - tfs[0]) / i
                current_delta_tf = (tfs[-1] - tfs[i - n_lust - 1]) / (n_lust)
                with open('output.txt', 'a') as file:
                    file.write(f'abs_delta_tf! {abs_delta_tf}\n')
                    file.write(f'curr_delta_tf! {current_delta_tf}\n')
            if i < (lust_count + n_lust):
                continue
            if current_delta_tf < abs_delta_tf * thresh_bond:
                with open('output.txt', 'a') as file:
                    file.write(f'slow growth rate detected! iter = {i}, alpha= {alpha}, ser_len = {maxlen}\n')
                lust_count = i + n_lust

                alphas = [alpha, low_param * alpha]
                if high_param * alpha < 1:
                    alphas.append(high_param * alpha)
                ser_lengths = [maxlen]
                if (high_param * maxlen) < max_ser_bond:
                    ser_lengths.append(int(high_param * maxlen))
                ser_lengths.append(int(low_param * maxlen))
                max_tf = -100000
                t_it = 0
                tfs_it = []
                tt = 0
                dds = 0
                alpha0 = 0
                print("slow start. tf = ", t.tf)
                mem = tfs[-1]
                for alpha1 in alphas:
                    for sl in ser_lengths:
                        tf_arr = []
                        t2 = copy.deepcopy(t)
                        maxtt = copy.deepcopy(maxt)
                        for j in range(n_lust):
                            dds, t2, tf_arr = splitius(sl, alpha1, tf_arr, t2, fc, maxtt)
                        with open('output1.txt', 'a') as file:
                            file.write(f'alt1 {alpha1} tf {t2.tf} {dds} len_arr {len(tf_arr)} {tf_arr}\n')

                        if dds > max_tf:
                            t_it = copy.deepcopy(t2)
                            maxlen = max(1, sl)
                            tfs_it = copy.deepcopy(tf_arr)
                            alpha0 = alpha1
                            max_tf = dds
                t = t_it
                i += n_lust
                alpha = alpha0
                for l1 in tfs_it:
                    tfs.append(l1)
                print("slow end. tf = ", t.tf)
                print(f'new params selected! iter = {i}, alpha= {alpha}, ser_len = {maxlen}\n')
                if abs(mem - tfs[-1]) < 0.0000001:
                    alpha = base_alpha
                    maxlen = base_len
                    print(
                        f'Поиск был неуспешен, восстановлены базовые параметры, alpha= {base_alpha}, ser_len = {base_len}\n')
                with open('output.txt', 'a') as file:
                    file.write(f'new params selected! iter = {i}, alpha= {alpha}, ser_len = {maxlen}\n')



    postfix(t, filename, s1, folder_path, ds)
    t.po=1
    initial_tf = t.calculate_tf(ds, p_tf, w, a, b, c, q, r, s, 0, ret_sum)
    t.po=0
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='merging.log', level=logging.INFO, filemode='w')
    logger.info('Started')
    num_merges = int(input(f'Total clusters: {t.maxclustid + 1}. input number of merges. :\n'))
    logger.info('num_merges: %d, Total clusters: %d', num_merges, t.clastnum)
    if num_merges > 0:
        t_merged = copy.deepcopy(t)
        initial_tf = t_merged.calculate_tf(ds, p_tf, w, a, b, c, q, r, s, 0, ret_sum)
        logger.info(f'Initial tf: {initial_tf}')

        min_tf_drop = float(input('input min fall value tf_new-tf_old / tf_old (0.8 equival to 20%  tf fall):\n'))
        min_tf_drop -= 1
        logger.info('Input min_tf_drop ' + str(min_tf_drop))
        logger.info('merging data')
        for i in range(num_merges):
            cluster1, cluster2, t_merged_new = merge_clusters_new(t_merged, i, ds, p_tf, w, a, b, c, q,
                                                                      r, 0, ret_sum)
            tf_stage = t_merged.calculate_tf(ds, p_tf, w, a, b, c, q, r, s, 0, ret_sum)
            logger.info(
                f'Stage {i}, tf={tf_stage}, clusters {cluster1} and {cluster2} merged')
            t_merged = t_merged_new
        tf_drop = (tf_stage - initial_tf) / abs(initial_tf)
        if tf_drop < min_tf_drop:
            t_merged = None
            print(f'tf after mergings: {tf_stage} ,Initial tf: {initial_tf}, old/new value: {1+tf_drop} , not accepted')
            logger.info(f'tf after mergings: {tf_stage} ,Initial tf: {initial_tf}, old/new value: {1+tf_drop} , not accepted')
        else:
            print(f'tf after mergings: {tf_stage} ,Initial tf: {initial_tf},  old/new value: {1+tf_drop} , accepted')
            logger.info(f'tf after mergings: {tf_stage} ,Initial tf: {initial_tf}, old/new value: {1+tf_drop} , accepted')
    else:
        t_merged = None

    if t_merged != None:
        print('save merging results')
        postfix(t_merged, 'merged_' + filename, s1, folder_path, ds, prefix='merged_')
    else:
        print('merging results is None')

if __name__ == '__main__':
    main()
