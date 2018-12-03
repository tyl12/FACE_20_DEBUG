//
// Created by chenchen on 12/07/18.
//

#include "model_sample.h"
#include <unistd.h>
#include <dirent.h>

char **label_cagerioes;

float VALID_PEOPLE[20][128];
float VALID_PEOPLE_SUM_SIGNAL[20];
float VALID_PEOPLE_FOUR_STAGE[20][4];
char PEOPLE_NAME[20][20];
int Total_valid_people=0;



extern std::mutex video_mutex;
extern cv::Mat bgr;
dp_image_box_t box_second[2];
dp_image_box_t BLOB_IMAGE_SIZE={0,1280,0,960};
unsigned short Resnet_Image_Buffer[224*224*3*50];


extern void fps_callback(int32_t *buffer_fps,void *param);
extern void blob_parse_callback(double *buffer_fps,void *param);
extern void video_callback(dp_img_t *img, void *param);
extern void cdk_result_model(void *result,void *param);
extern void cdk_two_result_model(void *result,void *param);


void test_hongxing_facenet(int argc, char *argv[])
{
    	int ret;
   	// const char *filename = "./model/face_detector_old.Blob";
   	const char *filename = "./model/face_detector.Blob";
    	const char *filename2 = "./model/face_recogntion.graph";

    	int blob_nums = 2; dp_blob_parm_t parms[2] = {{0,300,300,707*2 },{0,160,160,128*2}};  //模型尺寸改变了，图像尺寸变小，检测精度，效果想对于448尺寸，略有下降，但速度两个模型控制在了200ms以内目前来看是可以的
    	//int blob_nums = 2; dp_blob_parm_t parms[2] = {{0,448,448,1331*2 },{0,160,160,128*2}};
    	dp_netMean mean[2]={{0,0,0,255},{112.2917,112.2917,112.2917,59.7970}};
    	dp_set_blob_image_size(&BLOB_IMAGE_SIZE);
    	test_update_model_parems(blob_nums, parms);
    	dp_set_blob_mean_std(blob_nums,mean);
    	ret = dp_update_model(filename);
    	if (ret == 0) {
        	printf("Test dp_update_model(%s) sucessfully!\n", filename);
    	}
    	else {
        	printf("Test dp_update_model(%s) failed ! ret=%d\n", filename, ret);
    	}
    	ret = dp_update_model_2(filename2);
    	if (ret == 0) {
        	printf("Test dp_update_model_2(%s) sucessfully!\n", filename2);
    	}
    	else {
        	printf("Test dp_update_model_2(%s) failed ! ret=%d\n", filename2, ret);
    	}
    	DP_MODEL_NET net_1=DP_SSD_MOBILI_NET; //DP_TINY_YOLO_V2_FACE_NET;   //模型框架结构变化
    	dp_register_box_device_cb(cdk_result_model, &net_1);
    	DP_MODEL_NET net_2=DP_FACE_NET;
    	dp_register_second_box_device_cb(cdk_result_model,&net_2);
    	dp_register_video_frame_cb(video_callback, &net_1);
    	dp_register_fps_device_cb(fps_callback,&net_1);
    	dp_register_parse_blob_time_device_cb(blob_parse_callback,NULL);
    	ret = dp_start_camera_video();
    	if (ret == 0) {
        	printf("Test test_start_video successfully!\n");
    	}
    	else {
        	printf("Test test_start_video failed! ret=%d\n", ret);
    	}
	DIR  *dir;
	struct dirent *ptr;
	dir = opendir("face");
	int index=0;
	while((ptr = readdir(dir)) != NULL)
	{
		if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)
			continue;
		char filepath[30]="face/";
		strcat(filepath,ptr->d_name);
		printf("filepath:%s\n",filepath);
		FILE *fp=fopen(filepath,"rb");
		if(fp==NULL)
			continue;
		fread(VALID_PEOPLE[index],sizeof(float),128,fp);
		memcpy(PEOPLE_NAME[index],ptr->d_name,20);
		index++;
		fclose(fp);
	}
	
	for(int ii=0;ii<index;ii++)
	{
		for(int jj=0;jj<128;jj++)
		{
			printf(" %.3f  ",VALID_PEOPLE[ii][jj]);
			VALID_PEOPLE_SUM_SIGNAL[ii]+=VALID_PEOPLE[ii][jj];
			if(ii<32)
				VALID_PEOPLE_FOUR_STAGE[ii][0]+=VALID_PEOPLE[ii][jj];
			else if(ii<64)
				VALID_PEOPLE_FOUR_STAGE[ii][1]+=VALID_PEOPLE[ii][jj];
			else if(ii<96)
				VALID_PEOPLE_FOUR_STAGE[ii][2]+=VALID_PEOPLE[ii][jj];
			else if(ii<128)
				VALID_PEOPLE_FOUR_STAGE[ii][3]+=VALID_PEOPLE[ii][jj];
		}
			
		printf("\n");
	}
	Total_valid_people=index;
	printf("Total_valid_people:%d\n",Total_valid_people);
	closedir(dir);
    	const char *win_name = "video";
    	cv::namedWindow(win_name);
    	int key = -1;
    	for (;;) {
        	video_mutex.lock();
        	if (!bgr.empty())
            		cv::imshow(win_name, bgr);
        	video_mutex.unlock();
        	key = cv::waitKey(30);
    	}
    cv::destroyWindow(win_name);
}


