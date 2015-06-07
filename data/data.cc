#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string>
#include <algorithm>
#include <random>
#include <mutex>
#include <atomic>
#include <tbb/pipeline.h>
#include "blocks.pb.h"
typedef unsigned long long uint64;
typedef unsigned int       uint32;
#ifdef LEVEL1_DCACHE_LINESIZE
#define CACHE_LINE_SIZE LEVEL1_DCACHE_LINESIZE
#else
#define CACHE_LINE_SIZE 64
#endif
#define NUM_ITEM_IN_FLIGHT 8
#define NUM_BLOCKS 500
#define NTRAIN 1
typedef struct {
    int u_, v_, r_;
} Record;
Record* recs = (Record*)malloc(NTRAIN*sizeof(Record));

/* instead of 500K fread in this case,
   Can be optimized to only have 1 fread.
   Only necessary when InputFilter is slower than SgdFilter */
void write_str_to_message(char* read, char* write)
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
        if(ii%NUM_BLOCKS==0) {
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
  printf("#users = %d\n", count);
  google::protobuf::ShutdownProtobufLibrary();
}

int main(int argc, char** argv) {
  char *read=NULL, *write=NULL; 
  for(int i = 1; i < argc; i++) { 
	  if(!strcmp(argv[i], "-r"))            read = argv[++i];
	  else if(!strcmp(argv[i], "-w"))       write = argv[++i];
  }
  write_str_to_message(read, write);

  return 0;
}
