from collections import defaultdict
import codecs
import numpy as np

def read_sif(fpath, signs={'+/-':1,'+':2, '-':3}, sort=True, as_nx=False):
    dict_links = defaultdict(list)
    set_nodes = set()
    n2i = {}
    with codecs.open(fpath, "r", encoding="utf-8-sig") as fin:
        for line in fin:
            if line.isspace():
                continue

            items = line.strip().split()
            src = items[0]
            trg = items[2]
            sign = items[1]

            set_nodes.add(src)
            set_nodes.add(trg)
            int_sign = signs[sign]
            dict_links[src].append((trg, int_sign))
        # end of for
    # end of with

    if sort == True:
        list_nodes = sorted(set_nodes)
    else:
        list_nodes = list(set_nodes)

    N = len(set_nodes)
    adj = np.zeros((N, N), dtype=int)

    for isrc, name in enumerate(list_nodes):
        n2i[name] = isrc  # index of source
    # end of for
    for name_src in n2i:
        isrc = n2i[name_src]
        for name_trg, int_sign in dict_links[name_src]:
            itrg = n2i[name_trg]
            adj[itrg, isrc] = int_sign
        # end of for
    # end of for

    # print(set_nodes)

    return adj, n2i, list_nodes

def get_adj_index_given(fpath, node_nm_ls, signs={'+/-':1,'+':2, '-':3}):
    dict_links = defaultdict(list)
    set_nodes = set()

    node_name_ls = node_nm_ls
    name_to_idx = {}

    for idx, node_name in enumerate(node_name_ls):
        name_to_idx[node_name] = idx


    with codecs.open(fpath, "r", encoding="utf-8-sig") as fin:
        for line in fin:
            if line.isspace():
                continue

            items = line.strip().split()
            src = items[0]
            trg = items[2]
            sign = items[1]

            set_nodes.add(src)
            set_nodes.add(trg)
            int_sign = signs[sign]
            dict_links[src].append((trg, int_sign))
        # end of for
    # end of with

    N = len(node_name_ls)
    adj = np.zeros((N, N), dtype=int)


    for name_src in dict_links.keys():
        try:
            idx_src = name_to_idx[name_src]

            for name_trg, int_sign in dict_links[name_src]:
                idx_trg = name_to_idx[name_trg]
                adj[idx_trg, idx_src] = int_sign
        except:
            pass

    print(adj)

    return adj