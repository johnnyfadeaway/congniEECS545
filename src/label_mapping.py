# This program matches LPD ids and MSD ids to their labels

def main():
    cleansed_ids_obj = open("id_label_mapping/cleansed_ids.txt")
    data = cleansed_ids_obj.readlines()
    num_of_elements = len(data)
    clensed_LPD_ids = []
    clensed_MSD_ids = []
    for i in range(num_of_elements):
        clensed_LPD_ids.append(data[i][0:32])
        clensed_MSD_ids.append(data[i][36:54])
    cleansed_ids_obj.close()
    print(len(clensed_LPD_ids))
    print(len(clensed_MSD_ids))

    id_list_Country_obj = open("id_label_mapping/id_list_Country.txt")
    data = id_list_Country_obj.readlines()
    num_of_Country = len(data)
    id_list_Country = []
    for i in range(num_of_Country):
        id_list_Country.append(data[i][0:18])
    id_list_Country_obj.close()
    print(len(id_list_Country))

    id_list_Electronic_obj = open("id_label_mapping/id_list_Electronic.txt")
    data = id_list_Electronic_obj.readlines()
    num_of_Electronic = len(data)
    id_list_Electronic = []
    for i in range(num_of_Electronic):
        id_list_Electronic.append(data[i][0:18])
    id_list_Electronic_obj.close()
    print(len(id_list_Electronic))

    id_list_Jazz_obj = open("id_label_mapping/id_list_Jazz.txt")
    data = id_list_Jazz_obj.readlines()
    num_of_Jazz = len(data)
    id_list_Jazz = []
    for i in range(num_of_Jazz):
        id_list_Jazz.append(data[i][0:18])
    id_list_Jazz_obj.close()
    print(len(id_list_Jazz))

    id_list_Metal_obj = open("id_label_mapping/id_list_Metal.txt")
    data = id_list_Metal_obj.readlines()
    num_of_Metal = len(data)
    id_list_Metal = []
    for i in range(num_of_Metal):
        id_list_Metal.append(data[i][0:18])
    id_list_Metal_obj.close()
    print(len(id_list_Metal))

    id_list_Pop_obj = open("id_label_mapping/id_list_Pop.txt")
    data = id_list_Pop_obj.readlines()
    num_of_Pop = len(data)
    id_list_Pop = []
    for i in range(num_of_Pop):
        id_list_Pop.append(data[i][0:18])
    id_list_Pop_obj.close()
    print(len(id_list_Pop))

    id_list_RnB_obj = open("id_label_mapping/id_list_RnB.txt")
    data = id_list_RnB_obj.readlines()
    num_of_RnB = len(data)
    id_list_RnB = []
    for i in range(num_of_RnB):
        id_list_RnB.append(data[i][0:18])
    id_list_RnB_obj.close()
    print(len(id_list_RnB))

    id_list_Rock_obj = open("id_label_mapping/id_list_Rock.txt")
    data = id_list_Rock_obj.readlines()
    num_of_Rock = len(data)
    id_list_Rock = []
    for i in range(num_of_Rock):
        id_list_Rock.append(data[i][0:18])
    id_list_Rock_obj.close()
    print(len(id_list_Rock))

    MSD_id_map_genre = {}
    for i in range(num_of_elements):
        MSD_id_map_genre[clensed_MSD_ids[i]] = 0
    
    for i in range(num_of_Country):
        MSD_id_map_genre[id_list_Country[i]] = 1
        
    for i in range(num_of_Electronic):
        MSD_id_map_genre[id_list_Electronic[i]] = 2

    for i in range(num_of_Jazz):
        MSD_id_map_genre[id_list_Jazz[i]] = 3

    for i in range(num_of_Metal):
        MSD_id_map_genre[id_list_Metal[i]] = 4
    
    for i in range(num_of_Pop):
        MSD_id_map_genre[id_list_Pop[i]] = 5

    for i in range(num_of_RnB):
        MSD_id_map_genre[id_list_RnB[i]] = 6
    
    for i in range(num_of_Rock):
        MSD_id_map_genre[id_list_Rock[i]] = 7

    LPD_id_map_genre = dict(zip(clensed_LPD_ids,list(MSD_id_map_genre.values())))
    ## LPD_id_map_genre is the output dict

    ## print(LPD_id_map_genre["dfe2d11eec4c667e99ce399204276aa2"])






    
    

    







if __name__ == "__main__":
    main()








