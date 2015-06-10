#include <iostream>
#include <string.h>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <unordered_map>
#include <tuple>
#include <vector>
#include "blocks.pb.h"

typedef std::unordered_map<int, std::vector<std::pair<int, float>>> Dict;
typedef Dict::const_iterator DictIt;
typedef std::tuple<int, int, float> Tuple;
typedef unsigned long long uint64;
typedef unsigned int       uint32;
#define CACHE_LINE_SIZE 64

int block_size = 500;

void read_raw(char* file, std::vector<Tuple>& data) {
    FILE* fp = fopen(file, "r");
    int nn;
    int u,v,t;
    float r;
    fscanf(fp, "%d", &nn);
    data.resize(nn);
    for(int i=0; i<nn; i++) {
        fscanf(fp, "%d,%d,%f,%d", &u,&v,&r,&t);
        data.push_back(std::make_tuple(u,v,r));
    }
    std::random_shuffle(data.begin(), data.end());
    std::random_shuffle(data.begin(), data.end());
}

void write_by_dict(Dict& du, FILE* fp) {
    for(DictIt it(du.begin()); it!=du.end(); it++) {
        int u = it->first;
        fprintf(fp, "%d:\n", u);
        auto eles = it->second;
        for(std::vector<std::pair<int, float>>::iterator vit=eles.begin(); vit!=eles.end(); vit++) {
            int v = vit->first;
            float r = vit->second;
            fprintf(fp, "%d,%f\n", v, r);
        }
    }
}

void userwise(char* write, std::vector<Tuple>& data, int nb, int nresd, int bk) {
    int i;
    for(i=0; i<bk-1; i++) {
        Dict du;
        for(int j=i*nb; j<i*nb+nb; j++) {
            auto ele = data[j];
            int u = std::get<0>(ele);
            int v = std::get<1>(ele);
            float r = std::get<2>(ele);
            du[u].push_back(std::make_pair(v,r));
        }
        //write
        FILE* fp = fopen(write, "w");
        write_by_dict(du, fp);
        fclose(fp);
    }
    {
        Dict du;
        for(int j=i*nb; j<i*nb+nb+nresd; j++) {
            auto ele = data[j];
            int u = std::get<0>(ele);
            int v = std::get<1>(ele);
            float r = std::get<2>(ele);
            du[u].push_back(std::make_pair(v,r));
        }
        //write
        FILE* fp = fopen(write, "w");
        write_by_dict(du, fp);
        fclose(fp);
    }
}

void get_message(char* read, char* write)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    FILE* f_r = fopen(read, "r");
    FILE* f_w = fopen(write, "wb");
    char buf[CACHE_LINE_SIZE*10];
    int len, vid, count=0, ii=0, gc = 0;
    float rating;
    mf::Blocks blocks;
    mf::Block* block = NULL;
    mf::User* user = NULL;
    mf::User_Record* record = NULL;
    std::string uncompressed_buffer;
    uint32 uncompressed_size;
    while (fgets(buf,CACHE_LINE_SIZE*10,f_r)) {
        len = strlen(buf)-1;
        if(len==0)
        break;
        buf[len]='\0';
        if(buf[len-1]==':') {
            if(ii%block_size==0) {
                if(block) {
                    block->SerializeToString(&uncompressed_buffer);
                    uncompressed_size = uncompressed_buffer.size();
                    fwrite(&uncompressed_size, 1, sizeof(uncompressed_size), f_w);
                    fwrite(uncompressed_buffer.c_str(), 1, uncompressed_size, f_w);
                    count+=block->user_size();
                }
                block = blocks.add_block();
                user = block->add_user();
            }
            else {
                user = block->add_user();
            }
            ii++;
            user->set_uid(atoi(buf));
            continue;
        }
        record = user->add_record();
        sscanf(buf, "%d,%f", &vid, &rating);
        record->set_vid(vid);
        record->set_rating(rating);
    }
    block->SerializeToString(&uncompressed_buffer);
    uncompressed_size = uncompressed_buffer.size();
    fwrite(&uncompressed_size, 1, sizeof(uncompressed_size), f_w);
    fwrite(uncompressed_buffer.c_str(), 1, uncompressed_size, f_w);
    count+=block->user_size();
    fclose(f_r);
    fclose(f_w);
    google::protobuf::ShutdownProtobufLibrary();
}

void hint() {
    printf("-r         [input_file_name]\n");
    printf("-w         [output_file_name]\n");
    printf("--method   [userwise/protobuf]\n");
    printf("--split    [number_of_splits_for_rating_matrix]\thints: 1~10 splits are recommended\n");
    printf("--size     [number_of_users_in_each_block]\thints: 1 fread reads 1 block each time\n");
}

int main(int argc, char** argv) {
    char *read=NULL, *write=NULL;
    char *method=NULL;
    int bk=1;
    for(int i = 1; i < argc; i++) {
        if(!strcmp(argv[i], "-r"))            read = argv[++i];
        else if(!strcmp(argv[i], "-w"))       write = argv[++i];
        else if(!strcmp(argv[i], "--method")) method = argv[++i];
        else if(!strcmp(argv[i], "--split"))  bk = atoi(argv[++i]);
        else if(!strcmp(argv[i], "--size"))   block_size = atoi(argv[++i]);
        else {
            printf("unknown parameters.\n\n");
            hint();
            exit(1);
        }
    }
    if(read==NULL || write==NULL || method==NULL) {
        printf("Please at least indicate the input, output and method.\n\n");
        hint();
        exit(1);
    }

    if(!strcmp(method, "userwise")) {
        std::vector<Tuple> data;
        read_raw(read, data);
        int nn = data.size();
        int nb = nn/bk;
        int nresd = nn%bk;
        userwise(write, data, nb, nresd, bk);
    }
    else if(!strcmp(method, "protobuf")) {
        get_message(read, write);
    }
    else {
        exit(1);
    }
    return 0;
}
