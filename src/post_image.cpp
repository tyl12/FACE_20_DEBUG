//
// Created by chenchen on 11/07/18.
//
#include "post_image.h"
#include "detection_layer.h"

#include <math.h>
extern char **label_cagerioes;

extern float VALID_PEOPLE[20][128];
extern char PEOPLE_NAME[20][20];

extern float VALID_PEOPLE_FOUR_STAGE[20][4];

float resultfp32[128];

float result_stage[4];

extern int Total_valid_people;

void POST_IMAGE_MODEL::print_ssd_mobilet_result(void *fathomOutput,Box *box_demo,int *num_box_demo)
{
    u16* probabilities = (u16*)fathomOutput;
    unsigned int resultlen=707;
    float*resultfp32;
    resultfp32=(float*)malloc(resultlen * sizeof(*resultfp32));
    int img_width=1280;
    int img_height=960;
    for (u32 i = 0; i < resultlen; i++)
        resultfp32[i]= f16Tof32(probabilities[i]);
    int num_valid_boxes=int(resultfp32[0]);
    int index=0;
    for(int box_index=0;box_index<num_valid_boxes;box_index++)
    {
        int base_index=7*box_index+7;
        if(resultfp32[base_index+6]<0||resultfp32[base_index+6]>=1||resultfp32[base_index+5]<0||resultfp32[base_index+5]>=1||resultfp32[base_index+4]<0||resultfp32[base_index+4]>=1||resultfp32[base_index+3]<0||resultfp32[base_index+3]>=1||resultfp32[base_index+2]>=1||resultfp32[base_index+2]<0||resultfp32[base_index+1]<0)
        {
            continue;
        }
        box_demo[index].xmin=(int(resultfp32[base_index+3]*img_width)>0)?int(resultfp32[base_index+3]*img_width):0;
        box_demo[index].xmax=(int(resultfp32[base_index+5]*img_width)<img_width)?int(resultfp32[base_index+5]*img_width):img_width;
        box_demo[index].ymin=(int(resultfp32[base_index+4]*img_height)>0)?int(resultfp32[base_index+4]*img_height):0;
        box_demo[index].ymax=(int(resultfp32[base_index+6]*img_height)<img_height)?int(resultfp32[base_index+6]*img_height):img_height;
	if(box_demo[index].xmin-30>0)
		box_demo[index].xmin=box_demo[index].xmin-30;
	if(box_demo[index].xmax+30<img_width)
		box_demo[index].xmax=box_demo[index].xmax+30;
	if(box_demo[index].ymin-20>0)
		box_demo[index].ymin=box_demo[index].ymin-20;
	if(box_demo[index].ymax+30<img_height)
		box_demo[index].ymax=box_demo[index].ymax+30;
        index++;
    }
    *num_box_demo=index;
    free(resultfp32);
}

void POST_IMAGE_MODEL::print_age_net_result(void *fathomOutput)
{
    char *age_list[]={"0-2","4-6","8-12","15-20","25-32","38-43","48-53","60-100"};
    u16* probabilities=(u16*)fathomOutput;
    unsigned int resultlen=8;
    float *resultfp32=(float *)malloc(resultlen*sizeof(*resultfp32));
    for(u32 i=0;i<resultlen;i++)
        resultfp32[i]=f16Tof32(probabilities[i]);
    max_age pre_age = {0,0};
    for(int i=0;i<resultlen;i++)
    {
        if(pre_age.max_predected<=resultfp32[i])
        {
            pre_age.max_predected=resultfp32[i];
            pre_age.index=i;
        }
    }
    printf("the age range is:%s and the confidence of%.2f\n",age_list[pre_age.index],pre_age.max_predected);
    free(resultfp32);
}

void POST_IMAGE_MODEL::print_gender_net_result(void *fathomOutput)
{
    char *gender_list[]={"Male","Female"};
    u16* probabilities=(u16*)fathomOutput;
    unsigned int resultlen=2;
    float *resultfp32=(float *)malloc(resultlen*sizeof(*resultfp32));
    for(u32 i=0;i<resultlen;i++)
        resultfp32[i]=f16Tof32(probabilities[i]);
    max_age pre_gender = {0,0};
    for(int i=0;i<resultlen;i++)
    {
        if(pre_gender.max_predected<=resultfp32[i])
        {
            pre_gender.max_predected=resultfp32[i];
            pre_gender.index=i;
        }
    }
    printf("the age range is:%s and the confidence of %.2f\n",gender_list[pre_gender.index],pre_gender.max_predected);
    free(resultfp32);
}

void POST_IMAGE_MODEL::print_googlenet_result(void *fathomOutput)
{
    u16* probabilities=(u16*)fathomOutput;
    unsigned int resultlen=1000;
    float *resultfp32=(float *)malloc(resultlen*sizeof(*resultfp32));
    for(u32 i=0;i<resultlen;i++)
        resultfp32[i]=f16Tof32(probabilities[i]);
    float temp_predeiction=0.0;
    int index_temp=0;
    for(int i=0;i<5;i++)
    {
        temp_predeiction=resultfp32[i];
        index_temp=i;
        for(int j=i+1;j<resultlen;j++)
        {
            if(temp_predeiction<=resultfp32[j])
            {
                temp_predeiction=resultfp32[j];
                index_temp=j;
            }
        }
        resultfp32[index_temp]=resultfp32[i];
        resultfp32[i]=temp_predeiction;
        printf("prediction classes: %s and the probabilityes:%0.3f\n",label_cagerioes[index_temp],temp_predeiction);
    }
    free(resultfp32);
}

void POST_IMAGE_MODEL::print_resnet_result(void *fathomOutput)
{
    u16* probabilities=(u16*)fathomOutput;
    unsigned int resultlen=1000;
    float *resultfp32=(float *)malloc(resultlen*sizeof(*resultfp32));
    for(u32 i=0;i<resultlen;i++)
        resultfp32[i]=f16Tof32(probabilities[i]);
    float temp_predeiction=0.0;
    int index_temp=0;
    for(int i=0;i<5;i++)
    {
        temp_predeiction=resultfp32[i];
        index_temp=i;
        for(int j=i+1;j<resultlen;j++)
        {
            if(temp_predeiction<=resultfp32[j])
            {
                temp_predeiction=resultfp32[j];
                index_temp=j;
            }
        }
        resultfp32[index_temp]=resultfp32[i];
        resultfp32[i]=temp_predeiction;
        printf("prediction classes: %s and the probabilityes:%0.3f\n",label_cagerioes[index_temp],temp_predeiction);
    }
    free(resultfp32);
}

void POST_IMAGE_MODEL::print_squeezenet_result(void *fathomOutput)
{
    u16* probabilities=(u16*)fathomOutput;
    unsigned int resultlen=1000;
    float *resultfp32=(float *)malloc(resultlen*sizeof(*resultfp32));
    for(u32 i=0;i<resultlen;i++)
        resultfp32[i]=f16Tof32(probabilities[i]);
    float temp_predeiction=0.0;
    int index_temp=0;
    for(int i=0;i<5;i++)
    {
        temp_predeiction=resultfp32[i];
        index_temp=i;
        for(int j=i+1;j<resultlen;j++)
        {
            if(temp_predeiction<=resultfp32[j])
            {
                temp_predeiction=resultfp32[j];
                index_temp=j;
            }
        }
        resultfp32[index_temp]=resultfp32[i];
        resultfp32[i]=temp_predeiction;
        printf("prediction classes: %s and the probabilityes:%0.3f\n",label_cagerioes[index_temp],temp_predeiction);
    }
    free(resultfp32);
}

void POST_IMAGE_MODEL::print_tiny_yolov1_net_result(void *fathomOutput,Box *box_demo,int *num_box_demo)
{
    u16* probabilities = (u16*)fathomOutput;
    unsigned int resultlen=1470;
    YOLO_Result result[20];
    int result_num=0;
    float*resultfp32;
    resultfp32=(float*)malloc(resultlen * sizeof(*resultfp32));
    int img_width=1280;
    int img_height=960;
    for (u32 i = 0; i < resultlen; i++)
        resultfp32[i]= f16Tof32(probabilities[i]);
    interpret_output(resultfp32,result, &result_num, img_width, img_height,0.2);
    *num_box_demo=result_num;
    for (int i = 0; i < result_num; ++i)
    {
        printf("class:%s, x:%d, y:%d, width:%d, height:%d, probability:%.2f.\n",result[i].category,result[i].x,result[i].y,result[i].width,result[i].height,result[i].probability);
        box_demo[i].xmin=result[i].x;
        box_demo[i].xmax=result[i].x+result[i].width;
        if(box_demo[i].xmax>img_width)
            box_demo[i].xmax=img_width;
        box_demo[i].ymin=result[i].y;
        box_demo[i].ymax=result[i].y+result[i].height;
        if(box_demo[i].ymax>img_height)
            box_demo[i].ymax=img_height;
        memcpy(box_demo[i].category,result[i].category,20);
    }
    free(resultfp32);
}
void POST_IMAGE_MODEL::print_inception_result(void *fathomOutput)
{
    u16* probabilities = (u16*)fathomOutput;
    unsigned int resultlen=1001;
    float*resultfp32;
    resultfp32=(float*)malloc(resultlen * sizeof(*resultfp32));
    for (u32 i = 0; i < resultlen; i++)
        resultfp32[i]= f16Tof32(probabilities[i]);
    float temp_predeiction=0.0;
    int index_temp=0;
    for(int i=0;i<5;i++)
    {
        temp_predeiction=resultfp32[i];
        index_temp=i;
        for(int j=i+1;j<resultlen;j++)
        {
            if(temp_predeiction<=resultfp32[j])
            {
                temp_predeiction=resultfp32[j];
                index_temp=j;
            }
        }
        resultfp32[index_temp]=resultfp32[i];
        resultfp32[i]=temp_predeiction;
        printf("prediction classes: %s and the probabilityes:%0.3f\n",label_cagerioes[index_temp-2],temp_predeiction);
    }
    free(resultfp32);
}
void POST_IMAGE_MODEL::print_mnist_net_result(void* fathomOutput)
{
    u16* probabilities = (u16*)fathomOutput;
    char *labels[]={"0","1","2","3","4","5","6","7","8","9"};
    unsigned int resultlen=10;
    float*resultfp32;
    resultfp32=(float*)malloc(resultlen * sizeof(*resultfp32));
    for (u32 i = 0; i < resultlen; i++)
        resultfp32[i]= f16Tof32(probabilities[i]);
    float temp_predeiction=0.0;
    int index_temp=0;
    for(int i=0;i<5;i++)
    {
        temp_predeiction=resultfp32[i];
        index_temp=i;
        for(int j=i+1;j<resultlen;j++)
        {
            if(temp_predeiction<=resultfp32[j])
            {
                temp_predeiction=resultfp32[j];
                index_temp=j;
            }
        }
        resultfp32[index_temp]=resultfp32[i];
        resultfp32[i]=temp_predeiction;
        printf("prediction classes: %s and the probabilityes:%0.3f\n",labels[index_temp],temp_predeiction);
    }
    free(resultfp32);
}
void POST_IMAGE_MODEL::print_mobilinet_net_result(void *fathomOutput)
{
    u16* probabilities = (u16*)fathomOutput;
    unsigned int resultlen=1001;
    float*resultfp32;
    resultfp32=(float*)malloc(resultlen * sizeof(*resultfp32));
    for (u32 i = 0; i < resultlen; i++)
        resultfp32[i]= f16Tof32(probabilities[i]);
    float temp_predeiction=0.0;
    int index_temp=0;
    for(int i=0;i<5;i++)
    {
        temp_predeiction=resultfp32[i];
        index_temp=i;
        for(int j=i+1;j<resultlen;j++)
        {
            if(temp_predeiction<=resultfp32[j])
            {
                temp_predeiction=resultfp32[j];
                index_temp=j;
            }
        }
        resultfp32[index_temp]=resultfp32[i];
        resultfp32[i]=temp_predeiction;
        printf("prediction classes: %s and the probabilityes:%0.3f\n",label_cagerioes[index_temp],temp_predeiction);
    }
    free(resultfp32);
}
void POST_IMAGE_MODEL::print_tiny_yolov2_result(void *fathomOutput,Box *box_demo,int *num_box_demo)
{
    u16* probabilities = (u16*)fathomOutput;
    unsigned int resultlen=2430;
    std::vector<DetectedObject> results;
    int result_num=0;
    float* resultfp32=(float*)malloc(resultlen * sizeof(*resultfp32));
    float* new_data=(float*)malloc(resultlen * sizeof(*new_data));
    int img_width=1280;
    int img_height=960;
    for (u32 i = 0; i < resultlen; i++)
        resultfp32[i]= f16Tof32(probabilities[i]);
    reshape(resultfp32, new_data,resultlen,30,9,9);
    int blockwd = 9;
    int targetBlockwd =9;
    int classes = 1;
    float threshold = 0.5;
    float nms = 0.4;
    Region region_obj;
    region_obj.GetDetections(new_data,30,blockwd,blockwd,classes,img_width,img_height,threshold,nms,targetBlockwd,results);
    *num_box_demo= results.size();
    printf("results.size():%d\n",results.size());
    for (int i = 0; i <  results.size(); ++i)
    {
        printf("x:%d, y:%d, width:%d, height:%d, probability:%.2f.\n",results[i].left,results[i].top,(results[i].right-results[i].left),(results[i].bottom-results[i].top),results[i].confidence);
        box_demo[i].xmin=results[i].left;
	if(box_demo[i].xmin-15>0)
		box_demo[i].xmin=box_demo[i].xmin-15;
        box_demo[i].xmax=results[i].right;
        if(box_demo[i].xmax>img_width)
            box_demo[i].xmax=img_width;
	if(box_demo[i].xmax-15>0)
		box_demo[i].xmax=box_demo[i].xmax-15;
        box_demo[i].ymin=results[i].top;
        box_demo[i].ymax=results[i].bottom;
        if(box_demo[i].ymax>img_height)
            box_demo[i].ymax=img_height;
	if(box_demo[i].ymax-30>0)
		box_demo[i].ymax=box_demo[i].ymax-30;
    }
    free(resultfp32);
    free(new_data);
}

void POST_IMAGE_MODEL::print_tiny_yolov2_face_result(void *fathomOutput,std::vector<float>& probs, std::vector<cv::Rect>& boxes)
{
	u16* probabilities = (u16*)fathomOutput;
	unsigned int resultlen=1331;
	float* resultfp32=(float*)malloc(resultlen * sizeof(*resultfp32));
	for (u32 i = 0; i < resultlen; i++)
        resultfp32[i]= f16Tof32(probabilities[i]);
	get_detection_boxes(resultfp32, 1280,960, 0.2, probs, boxes);
	do_nms(boxes, probs, 1, 0.4);	
	for(int ii=0;ii<boxes.size();ii++)
		if(probs[ii]>0)
			printf("probes[%d]:%f ",ii,probs[ii]);
}


void POST_IMAGE_MODEL::print_alexnet_result(void *fathomOutput)
{
    u16* probabilities=(u16*)fathomOutput;
    unsigned int resultlen=1000;
    float *resultfp32=(float *)malloc(resultlen*sizeof(*resultfp32));
    for(u32 i=0;i<resultlen;i++)
        resultfp32[i]=f16Tof32(probabilities[i]);
    float temp_predeiction=0.0;
    int index_temp=0;
    for(int i=0;i<5;i++)
    {
        temp_predeiction=resultfp32[i];
        index_temp=i;
        for(int j=i+1;j<resultlen;j++)
        {
            if(temp_predeiction<=resultfp32[j])
            {
                temp_predeiction=resultfp32[j];
                index_temp=j;
            }
        }
        resultfp32[index_temp]=resultfp32[i];
        resultfp32[i]=temp_predeiction;
        printf("prediction classes: %s and the probabilityes:%0.3f\n",label_cagerioes[index_temp],temp_predeiction);
    }
    free(resultfp32);
}

void POST_IMAGE_MODEL::print_facenet_result(void *fathomOutput,char *peo_name)
{
    	u16* probabilities=(u16*)fathomOutput;
	
    	unsigned int resultlen=128;
    	
	float tmp_diff_sum=0.0f;
	float tmp_diff_stage=0.0f; 
	float tmp_diff_squre=0.0f;

	float Total_diff[20];
	float this_diff=0.0f;
	float tmp_mini_diff=0.0f;
	int tmp_mini_index=0;
    	for(u32 i=0;i<resultlen;i++)
    	{
        	resultfp32[i]=f16Tof32(probabilities[i]);
		printf("resultfp32[%d]:%.3f\n",i,resultfp32[i]);
		if(i<32)
			result_stage[0]+=resultfp32[i];
		if(i>=32&&i<64)
			result_stage[1]+=resultfp32[i];
		if(i>=64&&i<96)
			result_stage[2]+=resultfp32[i];
		if(i>=96&&i<128)
			result_stage[3]+=resultfp32[i];
    	}
	for(int ii=0;ii<4;ii++)
	 	printf("result_stage[%d]:%.3f ",ii,result_stage[ii]);
	printf("\n");

	for(int valid_people_index=0;valid_people_index<Total_valid_people;valid_people_index++)
	{
		float diff=0.0f;
		tmp_diff_squre=0.0f;
		tmp_diff_stage=0.0f;
		for(int output_index=0;output_index<resultlen;output_index++)
		{
			diff=VALID_PEOPLE[valid_people_index][output_index]-resultfp32[output_index];
			this_diff=pow(diff,2.0f);
			tmp_diff_squre+=this_diff;
		}
		printf("tmp_diff_squre:%.3f \n",tmp_diff_squre);
		/*for(int ii=0;ii<4;ii++)
		{
			diff=result_stage[ii]-VALID_PEOPLE_FOUR_STAGE[valid_people_index][ii];
			tmp_diff_stage+=pow(diff,2.0f);
		}
		printf("\ntmp_diff_stage:%.3f \n",tmp_diff_stage);*/
		Total_diff[valid_people_index]=tmp_diff_squre;

		printf("\nTotal_diff[%d]:%.3f\n",valid_people_index,Total_diff[valid_people_index]);
		if(valid_people_index==0)
		{
			tmp_mini_diff=Total_diff[0];
			tmp_mini_index=0;
		}
		else
		{
			if(tmp_mini_diff>Total_diff[valid_people_index])	
			{
				tmp_mini_diff=Total_diff[valid_people_index];
				tmp_mini_index=valid_people_index;
			}
		}
		
	}	
	printf("\nTotal_diff:%.3f\n",tmp_mini_diff);
	if(tmp_mini_diff>=-1.2&&tmp_mini_diff<=1.2)
		memcpy(peo_name,PEOPLE_NAME[tmp_mini_index],20);
	printf("detectored_name:%s\n",peo_name);
	for(int ii=0;ii<4;ii++)
		result_stage[ii]=0.0f;
}

