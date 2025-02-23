import numpy as np
import glob
import glm
import json
import os
import sys

# Change the file path to IEEE-13.glm
list_files = glob.glob("IEEE-13.glm")

l1 = glm.load(list_files[0])

l2 = {}
l2["name"] = "GLD"
l2["loglevel"] = 1
l2["coreType"] = "zmq"
l2["period"] = 10
l2["slow_responding"] = "True"
l2["broker"] = "127.0.0.1:6000"
l2["brokerPort"] = 6000
l2["broker_rank"] = 0
l2["endpoints"] = []

microgrids = {}
confi = {}

# Modified to handle IEEE-13 specific configurations
for i in l1["objects"]:
    if "configuration" in i["attributes"].keys():
        # Handle line, transformer, and regulator configurations
        config_type = i["attributes"]["configuration"].split("configuration")
        if len(config_type) > 1:
            tt = config_type[0].strip()
            if tt not in confi.keys():
                confi[tt] = []
            if "from" in i["attributes"].keys() and i["attributes"]["from"] not in confi[tt]:
                confi[tt].append(i["attributes"]["from"])
            if "to" in i["attributes"].keys() and i["attributes"]["to"] not in confi[tt]:
                confi[tt].append(i["attributes"]["to"])
    
    # Modified to handle IEEE-13's simpler topology
    if "from" in i["attributes"].keys():
        if len(microgrids.keys()) == 0:
            # Start with Node650 as the main source
            microgrids["SS_0"] = ["Node650", i["attributes"]["from"], i["attributes"]["to"]]
        else:
            found = False
            for xx in microgrids.keys():
                if i["attributes"]["from"] in microgrids[xx]:
                    microgrids[xx].append(i["attributes"]["to"])
                    found = True
                elif i["attributes"]["to"] in microgrids[xx]:
                    microgrids[xx].append(i["attributes"]["from"])
                    found = True
            if not found:
                microgrids["SS_"+str(len(microgrids.keys()))] = [i["attributes"]["from"], i["attributes"]["to"]]

# Modified to handle smaller number of groups for IEEE-13
num_group = min(int(sys.argv[1]), 3)  # Limit max groups to 3 for IEEE-13
tt = {}
kk = sorted(list(microgrids.keys()))
print("Size of keys: "+str(len(kk)))
for i in range(0,len(kk)):
    if "SS_"+str(i%num_group) not in tt.keys():
        tt["SS_"+str(i%num_group)] = []
    for x in microgrids[kk[i]]:
        tt["SS_"+str(i%num_group)].append(x)
microgrids = tt

res_micro = {}
for i in microgrids.keys():
    for c in microgrids[i]:
        res_micro[c] = i

# Modified to handle IEEE-13 specific types
res = {}
rr = {}
types = {}
types_ = {}
for i in l1["objects"]:
    if i["name"] not in types.keys():
        types[i["name"]] = []
    if "name" in i["attributes"].keys():
        if i["attributes"]["name"] not in types[i["name"]]:
            types[i["name"]].append(i["attributes"]["name"])

# Modified points collection for IEEE-13 specific components
points = {}
for i in l1["objects"]:
    if "name" in i["attributes"].keys():
        # Include regulator, switch, node, and load objects
        if i["attributes"]["name"] in res_micro.keys() and ("switch" in i["name"] or "regulator" in i["name"] or 
                                                          "node" in i["name"] or "load" in i["name"]):
            att = []
            for xx in types.keys():
                for ww in types[xx]:
                    if i["attributes"]["name"] in ww:
                        if res_micro[i["attributes"]["name"]] not in types_.keys():
                            types_[res_micro[i["attributes"]["name"]]] = {}
                        if xx not in types_[res_micro[i["attributes"]["name"]]].keys():
                            types_[res_micro[i["attributes"]["name"]]][xx] = []
                        if ww not in types_[res_micro[i["attributes"]["name"]]][xx]:
                            types_[res_micro[i["attributes"]["name"]]][xx].append(ww)
            
            for xx in list(i["attributes"].keys())[1:]:
                point_type = "ANALOG"
                if not any(term in xx.lower() for term in ["voltage", "current", "power", "tap"]):
                    point_type = "BINARY"
                if "flags" not in xx:
                    ss = {}
                    ss["name"] = res_micro[i["attributes"]["name"]]+"_"+i["attributes"]["name"]+"$"+xx
                    ss["type"] = "string"
                    ss["global"] = False
                    ss["info"] = "{\"%s\":\"%s\"}" % (i["attributes"]["name"], xx)
                    att.append(xx)
                    if res_micro[i["attributes"]["name"]] not in res.keys():
                        res[res_micro[i["attributes"]["name"]]] = []
                    res[res_micro[i["attributes"]["name"]]].append(ss)
                    if res_micro[i["attributes"]["name"]] not in points.keys():
                        points[res_micro[i["attributes"]["name"]]] = ""
                    if "ANALOG" in point_type:
                        points[res_micro[i["attributes"]["name"]]] += point_type +","+i["attributes"]["name"] + "$" + xx + ".real,0\n"
                        points[res_micro[i["attributes"]["name"]]] += point_type +","+i["attributes"]["name"] + "$" + xx + ".imag,0\n"
                    else:
                        points[res_micro[i["attributes"]["name"]]] += point_type +","+i["attributes"]["name"] + "$" + xx + ",0\n"
            if res_micro[i["attributes"]["name"]] not in rr.keys():
                rr[res_micro[i["attributes"]["name"]]] = {}
            rr[res_micro[i["attributes"]["name"]]][i["attributes"]["name"]] = att

# Create output directory if it doesn't exist
os.makedirs("../test_conf", exist_ok=True)

# Write points files
for i in sorted(list(points.keys())):
    with open("../test_conf/points_"+i+".csv", "w") as f:
        f.write(points[i])

# Modified grid configuration for IEEE-13
grid = {
    "Simulation": [{
        "SimTime": 35,
        "StartTime": 0.0,
        "PollReqFreq": 5,
        "includeMIM": 0,
        "UseDynTop": 0,
        "MonitorPerf": 1
    }],
    "microgrid": [],
    "DDoS": [{
        "NumberOfBots": 10,  # Reduced for IEEE-13
        "Active": 1,
        "Start": 1,
        "End": 11,
        "TimeOn": 10.0,
        "TimeOff": 0.0,
        "PacketSize": 100000,
        "Rate": "600000kb/s",
        "NodeType": ["MIM"],
        "NodeID": [2]
    }]
}

# Add microgrids
for c in sorted(list(types_.keys())):
    t = {
        "name": c,
        "dest": "ns3/"+c
    }
    for i in sorted(list(types_[c].keys())):
        t[i] = types_[c][i]
    grid["microgrid"].append(t)

# Modified MIM configuration for IEEE-13
grid["MIM"] = []
overview_MIM = {
    "NumberAttackers": 2,  # Reduced for IEEE-13
    "listMIM": "0,1"
}
grid["MIM"].append(overview_MIM)

# Add MIM attackers
for count in range(2):  # Reduced number of attackers
    t = {
        "name": f"MIM{count}",
        "attack_val": "TRIP",
        "real_val": "NA",
        "node_id": "switch1",  # Modified for IEEE-13
        "point_id": "status",
        "scenario_id": "b",
        "attack_type": "3",
        "Start": 120,
        "End": 180
    }
    grid["MIM"].append(t)

grid["controlCenter"] = {"name": "Monitor1"}

# Add endpoints
for x in sorted(list(res.keys())):
    for ss in res[x]:
        l2["endpoints"].append(ss)

    w = {
        "name": x,
        "type": "string",
        "global": False,
        "destination": "ns3/"+x
    }
    
    # Fixed the string formatting to avoid f-string escape issues
    info_parts = []
    for zz in rr[x].keys():
        values = '","'.join(rr[x][zz])  # Join the values with '","'
        info_parts.append('"{}":["{}"'.format(zz, values) + "]")
    
    w["info"] = "{" + ",".join(info_parts) + "}"
    l2["endpoints"].append(w)

# Modified topology configuration for IEEE-13
topology = {
    "Channel": [{
        "P2PDelay": "2ms",
        "CSMAdelay": "6560",
        "dataRate": "5Mbps",
        "jitterMin": 10,
        "jitterMax": 100,
        "WifiPropagationDelay": "ConstantSpeedPropagationDelayModel",
        "WifiRate": "DsssRate1Mbps",
        "WifiStandard": "80211b",
        "P2PRate": "250Mb/s"
    }],
    "Gridlayout": [{
        "MinX": 0,
        "MinY": 0,
        "DeltaX": 2000,  # Reduced for IEEE-13
        "DeltaY": 2000,  # Reduced for IEEE-13
        "GridWidth": 3,  # Modified for IEEE-13
        "LayoutType": "RowFirst",
        "SetPos": 1
    }],
    "5GSetup": [{
        "S1uLinkDelay": 1,
        "N1Delay": 0.01,
        "N2Delay": 0.01,
        "Srs": 10,  # Modified for IEEE-13
        "UeRow": len(types_.keys())/2 + 1,
        "UeCol": len(types_.keys()),
        "GnBRow": len(types_.keys()),
        "GnBCol": len(types_.keys())/2 + 1,
        "numUE": len(types_.keys()),
        "numEnb": len(types_.keys()),
        "scenario": "UMi-StreetCayon",
        "txPower": 40
    }],
    "Node": []
}

# Add nodes
for i in range(len(list(types_.keys()))):
    t = {
        "name": i,
        "connections": [i+1] if i+1 < len(types_.keys()) else [0],
        "UseCSMA": 0,
        "MTU": 1500,
        "UseWifi": 1,
        "x": i+2 if i%2 == 0 else i+8,
        "y": 5*(i+10) if i%2 == 0 else i*4,
        "error": "0.001"
    }
    topology["Node"].append(t)

# Write configuration files
with open("gridlabd_config.json", "w") as outfile:
    json.dump(l2, outfile, indent=8)

with open("grid.json", "w") as outfile:
    json.dump(grid, outfile, indent=8)

with open("topology.json", "w") as outfile:
    json.dump(topology, outfile, indent=8)