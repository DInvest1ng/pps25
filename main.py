from geting_mems import MEM_PARS


with open("ru_50k.txt", "r") as f:
    for line in f.readlines():
        mem = MEM_PARS()
        mem.save_bio(mem.get_links(line.split(" ß")))
