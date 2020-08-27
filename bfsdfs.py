from lxml import etree
import requests


def create_connections():
    url = 'https://www.bjsubway.com/station/zjgls/'
    r = requests.get(url)
    r.encoding = 'gbk'
    html = etree.HTML(r.text)
    line_block = html.xpath('//table[@height and @cellspacing]')

    line_name = []  # 线路名称
    line_station = {}  # 存放各个线路包含的站点:line_station['1号线']={s1,s2,...}
    station_dist = {}  # 相邻站点间距离信息:station_dist[s1][s2] = 1000m
    for line in line_block:
        # line_name:
        text = str(line.xpath('./thead//td[@colspan]/text()'))
        name = text.split('相邻')[0].replace("['", "")
        line_name.append(name)
        block = line.xpath('./tbody')
        stations = block[0].xpath('./tr')  # 包含:{站点-站点，距离，上行/下行} 的list

        line_station[name] = set()
        for pair in stations:
            # line_station:
            s1, s2 = pair.xpath('./th/text()')[0].split('——')
            distance = int(pair.xpath('./td/text()')[0])
            line_station[name].add(s1)
            line_station[name].add(s2)
            # station_dist:
            if s1 in station_dist:
                station_dist[s1] = dict([(s2, distance)] + list(station_dist[s1].items()))
            else:
                station_dist[s1] = {s2: distance}
            if s2 in station_dist:
                station_dist[s2] = dict([(s1, distance)] + list(station_dist[s2].items()))
            else:
                station_dist[s2] = {s1: distance}

    return station_dist


def search(graph, start, dest, mode='bfs'):
    if not isinstance(graph, dict):
        raise TypeError()
    if mode != 'bfs' and mode != 'dfs':
        raise ValueError()

    candidate = [[start]]
    success = []
    while not success:
        head = candidate[0]
        candidate.remove(candidate[0])
        next = graph[head[-1]].keys()
        new_route = []
        for next_one in next:
            if next_one == dest:
                tmp = head.copy()
                tmp.append(next_one)
                success.append(tmp)
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

                # 搜索完毕
    return success


if __name__ == '__main__':
    # 收集临近站点资料的网页和官网地图信息不一致。。。
    graph = create_connections()
    # rst = search(graph, '天安门东', '北新桥', mode='bfs')
    rst = search(graph, '长春桥', '磁器口', mode='dfs')
    print(rst)