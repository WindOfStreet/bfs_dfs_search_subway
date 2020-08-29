from lxml import etree
import requests
import pickle
import pandas as pd

class SubwaySearch:
    # 提供地铁查询功能，新建对象须运行create_connections()建立数据，或运行load_data()加载数据。之后可以使用查询功能search_best()和search()
    def __init__(self, ):
        self.station_dist = {}   # 相邻站点间距离信息:station_dist[s1][s2] = 1000m
        self.line_station = {}   # 存放各个线路包含的站点:line_station['1号线']={s1,s2,...}
        self.mean_dist = 0       # 相邻站点间平均距离:结果为1200m
        self.transfer_lines = []  # 换乘计划

    def create_connections(self, connect_save_path, stations_save_path):
        # 计算站点间距离和邻居关系
        # 计算各地铁线路包含站点信息
        url = 'https://www.bjsubway.com/station/zjgls/'
        r = requests.get(url)
        r.encoding = 'gbk'
        html = etree.HTML(r.text)
        line_block = html.xpath('//table[@height and @cellspacing]')

        line_name = []  # 线路名称
        for line in line_block:
            # line_name:
            text = str(line.xpath('./thead//td[@colspan]/text()'))
            name = text.split('相邻')[0].replace("['", "")
            line_name.append(name)
            block = line.xpath('./tbody')
            stations = block[0].xpath('./tr')  # 包含:{站点-站点，距离，上行/下行} 的list

            self.line_station[name] = set()
            for pair in stations:
                # line_station:
                s1, s2 = pair.xpath('./th/text()')[0].split('——')
                distance = int(pair.xpath('./td/text()')[0])
                self.line_station[name].add(s1)
                self.line_station[name].add(s2)
                # station_dist:
                if s1 in self.station_dist:
                    self.station_dist[s1] = dict([(s2, distance)] + list(self.station_dist[s1].items()))
                else:
                    self.station_dist[s1] = {s2: distance}
                if s2 in self.station_dist:
                    self.station_dist[s2] = dict([(s1, distance)] + list(self.station_dist[s2].items()))
                else:
                    self.station_dist[s2] = {s1: distance}

        # 计算站间平均距离
        cnt = 0
        total_dist = 0
        for item in self.station_dist.values():
            cnt += len(item)
            for val in item.values():
                total_dist += val
        self.mean_dist = total_dist / cnt

        # 保存数据
        f = open(connect_save_path, 'wb')
        pickle.dump(self.station_dist, f)
        f.close()
        f = open(stations_save_path, 'wb')
        pickle.dump(self.line_station, f)
        f.close()
        return self.station_dist

    def load_data(self, station_dist_path, line_station_path):
        # 分别加载站点间距离信息和地铁线路的包含站点信息
        self.station_dist = self.load_dict(station_dist_path)
        self.line_station = self.load_dict(line_station_path)

    @staticmethod
    def load_dict(path):
        f = open(path, 'rb')
        data = pickle.load(f)
        f.close()
        return data

    def search(self, start, destination, mode='bfs'):
        # 给定起止站点，输出深度优先或广度优先搜索的结果
        if mode != 'bfs' and mode != 'dfs':
            raise ValueError()

        candidate = [[start]]
        success = []
        while not success:
            head = candidate[0]
            candidate.remove(candidate[0])
            next_list = self.station_dist[head[-1]].keys()
            new_route = []
            for next_one in next_list:
                if next_one == destination:
                    tmp = head.copy()
                    tmp.append(next_one)
                    success = tmp
                    return success
                elif next_one in head:
                    continue
                else:
                    tmp = head.copy()
                    tmp.append(next_one)
                    new_route.append(tmp)

            if len(new_route) == 0:
                continue
            else:
                if mode == 'bfs':
                    candidate.extend(new_route)
                elif mode == 'dfs':
                    tmp = new_route.copy()
                    tmp.extend(candidate)
                    candidate = tmp.copy()

        return success

    def __calc_dist(self, route):
        # 计算输入路线的总里程：米
        dist = 0
        for i in range(len(route)-1):
            dist += self.station_dist[route[i]][route[i+1]]
        return dist

    def __calc_transfers(self, route):
        # 计算输入线路的换乘次数并返回
        # 统计线路中各个站点都属于哪些线路，并建立dataframe
        table = pd.DataFrame(data=0, index=self.line_station.keys(), columns=route)
        for station in route:
            for line_name in self.line_station.keys():
                if station in self.line_station[line_name]:
                    table.loc[line_name, station] = 1
        # 对于table中的每一行（一条地铁线路），若其中包含两个连续的1，则route换乘了此地铁线路
        # 当前暂忽略换乘了某线路两次的情形，即类似1号线-2号线-8号线-1号线。认为换乘了1次1号线
        station_cnt = len(route)
        count = pd.DataFrame(data=0, index=self.line_station.keys(), columns=range(station_cnt-1))
        for i in range(len(count.index.to_list())):
            for j in range(station_cnt-1):
                count.iloc[i, j] = table.iloc[i, j] + table.iloc[i, j+1]

        '''
        # 对于count的每一行（每一条地铁线路），若此行包含元素=2的值，则route线路换乘了此地铁线路
        transfer_lines = []
        for line_name in count.index:
            if 2 in count.loc[line_name, :].values:
                transfer_lines.append(line_name)
        '''
        # 按列取出换乘的线路，这样保留了先后顺序
        stop_list = []
        self.transfer_lines.clear()
        for col in count.columns:
            for idx in count.index:
                if count.loc[idx, col] == 2 and idx not in stop_list:
                    stop_list.append(idx)
                    self.transfer_lines.append(idx)

        return len(self.transfer_lines)-1

    def __compare(self, dist0, tran0, line_new, mode):
        # 比较新线路是否在给定的模式下更优，更优则返回true
        # dist0:当前最短距离
        # tran0：当前最小换乘次数
        # 待计算的线路
        # mode：模式：距离最短/换乘最少
        dist1 = self.__calc_dist(line_new)  # 新线路行驶距离
        tran1 = self.__calc_transfers(line_new)  # 新线路换乘次数
        rst = False  # 返回结果

        # 换乘次数判据：在最短距离模式下，若新路线的换乘次数超过一定数值，则同样认定新线路不是更优的
        tran_criterion = 0
        # 总行驶距离判据：在最少换乘模式下，若新线路行驶距离超过一定数值，则同样认定新线路不是更优的
        dist_criterion = 0
        if dist0 < 5000:
            # 5000m以内时，距离最短线路可以多换乘1次
            tran_criterion = 1
            # 5000m以内，少换乘线路可以多行驶3000m
            dist_criterion = 3000
        elif dist0 < 10000:
            tran_criterion = 2
            dist_criterion = 5000
        else:
            tran_criterion = 3
            dist_criterion = 6000

        if mode == 'min_dist':
            if dist1 < dist0 and tran1 - tran0 < tran_criterion:
                return True
            else:
                return False
        else:
            if tran1 < tran0 and dist1 - dist0 < dist_criterion:
                #print(tran0,dist0,tran1,dist1)
                return True
            else:
                return False

    def search_best(self, start, destination, mode='min_dist'):
        # 给定起止站点，查询最短距离或最少换乘的线路
        # start:起始站点
        # destination:终止站点
        # mode：模式：min_dist-最短距离，min_transfer-最少换乘
        # return:
        # success: 最优路线
        # curt_dist：路线距离
        # curt_trans：换乘次数
        # self.transfer_lines：按序换乘的车次

        if mode != 'min_dist' and mode != 'min_transfer':
            raise ValueError('must be')
        # 先以bfs计算出一个解，其他解和它比较，不满足的可以提前移除
        # 计算bfs解及其距离、换乘次数
        success = self.search(start, destination, 'bfs')
        curt_dist = self.__calc_dist(success)
        curt_trans = self.__calc_transfers(success)
        # 候选列表中有待搜索路线，则继续循环
        candidate = [[start]]
        while candidate:
            head = candidate[0]
            candidate.remove(candidate[0])
            next_list = self.station_dist[head[-1]].keys()
            new_route = []
            for next_one in next_list:
                if next_one == destination:
                    tmp = head.copy()
                    tmp.append(next_one)
                    if self.__compare(curt_dist, curt_trans, tmp, mode):
                        success = tmp.copy()
                        curt_dist = self.__calc_dist(success)
                        curt_trans = self.__calc_transfers(success)
                elif next_one in head:
                    continue
                else:
                    tmp = head.copy()
                    tmp.append(next_one)
                    if self.__compare(curt_dist, curt_trans, tmp, mode):
                        new_route.append(tmp)

            if len(new_route) == 0:
                continue
            else:
                candidate.extend(new_route)
        # 更新一下换乘车次列表
        self.__calc_transfers(success)
        return success, curt_dist, curt_trans, self.transfer_lines


if __name__ == '__main__':
    # 收集临近站点资料的网页和官网地图信息不一致，地图包含的线路日期更新
    station_connect_path = 'data/connections.dict'
    line_stations_path = 'data/stations.dict'


    # 训练模型
    # subway = SubwaySearch()
    # subway.create_connections(station_connect_path, line_stations_path)

    # 深度优先/广度优先
    # rst = search(graph, '天安门东', '北新桥', mode='bfs')
    # rst = subway.search('长春桥', '磁器口', mode='bfs')

    # 最短路径/最少换乘
    # rst1, dist = subway.search_best('和平门', '雍和宫', 'min_dist')
    # rst2, trans = subway.search_best('和平门', '雍和宫', 'min_transfer')
    # print('最短线路：{}，距离为{}米。'.format(rst1, dist))
    # print('最少换乘：{}，换乘次数为{}次。'.format(rst2, trans))

    # 从文件加载模型
    subway1 = SubwaySearch()
    subway1.load_data(station_connect_path, line_stations_path)
    rst, dist, trans, trans_lines = subway1.search_best('公主坟', '什刹海', 'min_transfer')
    print('线路：{}，距离为{}米。\n换乘{}次，换乘车次为{}。'.format(rst, dist, trans, trans_lines))

