import copy

def merge(cluster1, cluster2, clusters):
    merged_clusters = clusters.copy()
    for node, cluster in merged_clusters.items():
        if cluster == cluster2:
            merged_clusters[node] = cluster1
    return merged_clusters

def form_table_clusters(clusters_merged):
    clusters_merged_t = {}
    for node in clusters_merged.keys():
        cl = clusters_merged[node]
        if cl not in clusters_merged_t.keys():
            clusters_merged_t[cl] = {node}
        else:
            nodes = clusters_merged_t[cl]
            nodes.add(node)
            clusters_merged_t[cl] = nodes
    return clusters_merged_t

def reform_clusters_dict(t_clusters):
    clusters={}
    for cl in t_clusters.keys():
        for node in t_clusters[cl]:
            clusters[node]=cl
    return clusters

def find_neighboring_clusters(clusters, nodes):
    neighboring_clusters = set()
    for node, neighbors in nodes.items():
        for neighbor in neighbors:
            if clusters[node] != clusters[neighbor]:
                neighboring_clusters.add((clusters[node], clusters[neighbor]))
    return neighboring_clusters

def merge_clusters(graph, min_tf_drop,ns, ds, p_tf, w, a, b, c, q, r, wrp, ret_sum):
    prev_tf = graph.calculate_tf(ds, p_tf, w, a, b, c, q, r, 0, ret_sum)
    clusters=reform_clusters_dict(graph.clusters)
    best_merge = None
    best_tf = prev_tf
    btd = -1000
    graph1 = copy.deepcopy(graph)
    for cluster1, cluster2 in find_neighboring_clusters(clusters, graph1.edgemap):
        merged_clusters = form_table_clusters(merge(cluster1, cluster2, clusters))
        graph1.set_new_clusters(merged_clusters)
        tf = graph1.calculate_tf(ds, p_tf, w, a, b, c, q, r, 0, ret_sum)
        tf_drop = (tf - prev_tf) / abs(prev_tf)
        if tf_drop > min_tf_drop:
            if tf > best_tf:
                best_tf = tf
                best_merge = (cluster1, cluster2, merged_clusters)
                btd = tf_drop

    if best_merge is None:
        return None, None, None, None
    else:
        cluster1, cluster2, clusters = best_merge
        graph1.set_new_clusters(clusters)
        return cluster1, cluster2, btd, graph1

def merge_clusters_new(graph,ns,ds, p_tf, w, a, b, c, q, r, wrp, ret_sum):
    prev_tf = graph.calculate_tf(ds, p_tf, w, a, b, c, q, r, 0, ret_sum)
    clusters=reform_clusters_dict(graph.clusters)
    best_merge = None
    best_tf = -100000000000000
    graph1 = copy.deepcopy(graph)
    for cluster1, cluster2 in find_neighboring_clusters(clusters, graph1.edgemap):
        merged_clusters = form_table_clusters(merge(cluster1, cluster2, clusters))
        graph1.set_new_clusters(merged_clusters)
        tf = graph1.calculate_tf(ds, p_tf, w, a, b, c, q, r, 0, ret_sum)
        if tf > best_tf:
            best_tf = tf
            best_merge = (cluster1, cluster2, merged_clusters)

    if best_merge is None:
        print('omg mergenone')
        return None, None, None
    else:
        cluster1, cluster2, clusters = best_merge
        graph1.set_new_clusters(clusters)
        return cluster1, cluster2, graph1
