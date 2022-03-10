from operator import getitem
from loader import TempoSet
import numpy as np



def trim(song_mat,trim_length=50):
    """
    arg: the matrix for the entire song
    return: 512*512 trimed version, with the highest 0-norm value
    """

    time_stamp = song_mat.shape[0]
    num_of_notes = song_mat.shape[1]
    temp = np.empty((trim_length,num_of_notes))
    trimed_mat = song_mat[0:trim_length]
    niters = time_stamp - trim_length
    for i in range(niters):
        # compare the 0_norm of the ith window with the (i+1)th window
        temp = song_mat[i+1:(i+trim_length+1)]
        temp_0_norm = len(np.where(temp != 0))
        trimed_mat_0_norm = len(np.where(trimed_mat != 0))
        if temp_0_norm >= trimed_mat_0_norm:
            trimed_mat = temp.copy()
            
    return trimed_mat


def main():
    '''
        ojective of this main function:
        ** create a .npz file containing
            1. 7 3D arrays for each genre: M*N*num of songs in that genre
            2. 7 1D arrays containing the index of each songs in each genre, which can be later used to get drumtracks for that song
    '''
    M = 50
    N = 512
    data_dir = "../data/lpd_5/lpd_5_cleansed"
    loader = TempoSet()
    loader.load(data_dir)
    num_of_total_songs = len(loader)
    genre_1_mat = np.zeros((M,N,1))
    genre_2_mat = np.zeros((M,N,1))
    genre_3_mat = np.zeros((M,N,1))
    genre_4_mat = np.zeros((M,N,1))
    genre_5_mat = np.zeros((M,N,1))
    genre_6_mat = np.zeros((M,N,1))
    genre_7_mat = np.zeros((M,N,1))
    temp_mat = np.zeros((M,N,1))
    genre_1_list = []
    genre_2_list = []
    genre_3_list = []
    genre_4_list = []
    genre_5_list = []
    genre_6_list = []
    genre_7_list = []

    
    for i in range(num_of_total_songs):
        # a: other 4 instruments
        # b: a dict contraining "label(one hot vector)" and "durmtrack"

        a,b = getitem(loader,i)
        a = np.array(a)
        genre = np.array(b["genre"])
        print(i)
        if genre[0] == 1:
            continue
        else:
            if genre[1] == 1:
                if genre_1_mat.shape[2] >= 98:
                    continue
                else:
                    genre_1_list.append(i)
                    temp_mat = trim(a)
                    temp_mat = temp_mat[:,:,np.newaxis] # expand this matrix to 3D array
                    genre_1_mat = np.concatenate((genre_1_mat,temp_mat),axis=2)
                    continue

            elif genre[2] == 1:
                if genre_2_mat.shape[2] >= 98:
                    continue
                else:
                    genre_2_list.append(i)
                    temp_mat = trim(a)
                    temp_mat = temp_mat[:,:,np.newaxis] # expand this matrix to 3D array
                    genre_2_mat = np.concatenate((genre_2_mat,temp_mat),axis=2)
                    continue
            elif genre[3] == 1:
                if genre_3_mat.shape[2] >= 98:
                    continue
                else:
                    genre_3_list.append(i)
                    temp_mat = trim(a)
                    temp_mat = temp_mat[:,:,np.newaxis] # expand this matrix to 3D array
                    genre_3_mat = np.concatenate((genre_3_mat,temp_mat),axis=2)
                    continue
            elif genre[4] == 1:
                if genre_4_mat.shape[2] >= 98:
                    continue
                else:
                    genre_4_list.append(i)
                    temp_mat = trim(a)
                    temp_mat = temp_mat[:,:,np.newaxis] # expand this matrix to 3D array
                    genre_4_mat = np.concatenate((genre_4_mat,temp_mat),axis=2)
                    continue
            elif genre[5] == 1:
                if genre_5_mat.shape[2] >= 98:
                    continue
                else:
                    genre_5_list.append(i)
                    temp_mat = trim(a)
                    temp_mat = temp_mat[:,:,np.newaxis] # expand this matrix to 3D array
                    genre_5_mat = np.concatenate((genre_5_mat,temp_mat),axis=2)
                    continue
            elif genre[6] == 1:
                if genre_6_mat.shape[2] >= 98:
                    continue
                else:
                    genre_6_list.append(i)
                    temp_mat = trim(a)
                    temp_mat = temp_mat[:,:,np.newaxis] # expand this matrix to 3D array
                    genre_6_mat = np.concatenate((genre_6_mat,temp_mat),axis=2)
                    continue
            elif genre[7] == 1:
                if genre_7_mat.shape[2] >= 98:
                    continue
                else:
                    genre_7_list.append(i)
                    temp_mat = trim(a)
                    temp_mat = temp_mat[:,:,np.newaxis] # expand this matrix to 3D array
                    genre_7_mat = np.concatenate((genre_7_mat,temp_mat),axis=2)
                    continue
                
    

    genre_1_list = np.array(genre_1_list)
    genre_1_list = np.array(genre_1_list)
    genre_1_list = np.array(genre_1_list)
    genre_1_list = np.array(genre_1_list)
    genre_1_list = np.array(genre_1_list)
    genre_1_list = np.array(genre_1_list)
    genre_1_list = np.array(genre_1_list)

    np.savez("subs_learn_data_50.npz",genre_1_list=genre_1_list,genre_2_list=genre_2_list,genre_3_list=genre_3_list,genre_4_list=genre_4_list,genre_5_list=genre_5_list,genre_6_list=genre_6_list,genre_7_list=genre_7_list,genre_1_mat=genre_1_mat,genre_2_mat=genre_2_mat,genre_3_mat=genre_3_mat,genre_4_mat=genre_4_mat,genre_5_mat=genre_5_mat,genre_6_mat=genre_6_mat,genre_7_mat=genre_7_mat)

    print(genre_1_mat.shape)
    print(genre_2_mat.shape)
    print(genre_3_mat.shape)
    print(genre_4_mat.shape)
    print(genre_5_mat.shape)
    print(genre_6_mat.shape)
    print(genre_7_mat.shape)

    print(len(genre_1_list))
    print(len(genre_2_list))
    print(len(genre_3_list))
    print(len(genre_4_list))
    print(len(genre_5_list))
    print(len(genre_6_list))
    print(len(genre_7_list))
    


if __name__ == "__main__":
    main()




