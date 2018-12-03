//
// Created by chenchen on 12/07/18.
//
#include "post_image.h"
POST_IMAGE_MODEL sample_model;
double Sum_blob_parse_time=0;
double blob_parse_stage[400];
std::string blob_parse;
int blob_stage_index;

int32_t fps;

std::mutex video_mutex;
cv::Mat bgr;

int num_box_demo=0;
Box box_demo[100];

char detector_people_name[20];

extern void floattofp16(unsigned char *dst, float *src, unsigned nelem);
unsigned short tempbuffer[160*160*3];
float imgbuffer[160*160*3];



//帧率回调函数
void fps_callback(int32_t *buffer_fps,void *param)
{
    fps=*(buffer_fps);
}

//解析模型时间回调函数
void blob_parse_callback(double *buffer_fps,void *param)
{
    for(int stage=0;stage<200;stage++)
    {
        blob_parse_stage[stage]=buffer_fps[stage*2+0];
        blob_stage_index=buffer_fps[stage*2+1];
        Sum_blob_parse_time+=blob_parse_stage[stage];
        std::ostringstream   ostr;
        ostr<<"the"<<stage<<"stage parse spending"<<blob_parse_stage[stage]<<"ms,and optType:"<<OP_NAMES[blob_stage_index]<<"\n";
        blob_parse.append(ostr.str());
        if((stage+1)<200)
        {
            if(buffer_fps[(stage+1)*2+0]==0)
                break;
        }
    }     std::ostringstream   ostr;
    ostr <<"the total spending "<<Sum_blob_parse_time<<" ms\n";
    blob_parse.append(ostr.str());
    Sum_blob_parse_time=0;
}

//视频帧回调函数
void video_callback(dp_img_t *img, void *param)
{
    cv::Mat myuv(img->height + img->height / 2, img->width, CV_8UC1, img->img);
    video_mutex.lock();
    cvtColor(myuv, bgr, CV_YUV2BGR_I420, 0);
    DP_MODEL_NET model=*((DP_MODEL_NET*)param);
    switch (model)
    {
        case DP_TINI_YOLO_NET:
        {
            for(int i=0;i<num_box_demo;i++)
            {
                cv::rectangle(bgr,cvPoint(box_demo[i].xmin,box_demo[i].ymin),cvPoint(box_demo[i].xmax,box_demo[i].ymax),CV_RGB(0, 255, 0), 2);
                cv::putText(bgr,box_demo[i].category,cvPoint(box_demo[i].xmin,box_demo[i].ymin),cv::FONT_HERSHEY_PLAIN,2,CV_RGB(0, 255, 0),2,8);
            }
            std::string buffer="fps:"+std::to_string(fps);
            cv::putText(bgr,buffer.c_str(),cvPoint(40,40),cv::FONT_HERSHEY_PLAIN,2,CV_RGB(0, 255, 0),2,8);
            break;
        }
        case DP_SSD_MOBILI_NET:
        {
            for(int i=0;i<num_box_demo;i++)
            {
                cv::rectangle(bgr,cvPoint(box_demo[i].xmin,box_demo[i].ymin),cvPoint(box_demo[i].xmax,box_demo[i].ymax),CV_RGB(0, 255, 0), 2);
				memcpy(box_demo[i].category,detector_people_name,20);
                cv::putText(bgr,box_demo[i].category,cvPoint(box_demo[i].xmin,box_demo[i].ymin),cv::FONT_HERSHEY_PLAIN,2,CV_RGB(0, 255, 0),2,8);
            }
            std::string buffer="fps:"+std::to_string(fps);
            cv::putText(bgr,buffer.c_str(),cvPoint(40,40),cv::FONT_HERSHEY_PLAIN,2,CV_RGB(0, 255, 0),2,8);
            break;
        }
		case DP_TINY_YOLO_V2_FACE_NET:
        {
            for(int i=0;i<num_box_demo;i++)
            {
                cv::rectangle(bgr,cvPoint(box_demo[i].xmin,box_demo[i].ymin),cvPoint(box_demo[i].xmax,box_demo[i].ymax),CV_RGB(0, 255, 0), 2);
				memcpy(box_demo[i].category,detector_people_name,20);
                cv::putText(bgr,box_demo[i].category,cvPoint(box_demo[i].xmin,box_demo[i].ymin),cv::FONT_HERSHEY_PLAIN,2,CV_RGB(0, 255, 0),2,8);
            }
            std::string buffer="fps:"+std::to_string(fps);
            cv::putText(bgr,buffer.c_str(),cvPoint(40,40),cv::FONT_HERSHEY_PLAIN,2,CV_RGB(0, 255, 0),2,8);
            break;
        }
        case DP_TINY_YOLO_V2_NET:
        {
            for(int i=0;i<num_box_demo;i++)
            {
                cv::rectangle(bgr,cvPoint(box_demo[i].xmin,box_demo[i].ymin),cvPoint(box_demo[i].xmax,box_demo[i].ymax),CV_RGB(0, 255, 0), 2);
		memcpy(box_demo[i].category,detector_people_name,20);
                cv::putText(bgr,box_demo[i].category,cvPoint(box_demo[i].xmin,box_demo[i].ymin),cv::FONT_HERSHEY_PLAIN,2,CV_RGB(0, 255, 0),2,8);
            }
            std::string buffer="fps:"+std::to_string(fps);
            cv::putText(bgr,buffer.c_str(),cvPoint(40,40),cv::FONT_HERSHEY_PLAIN,2,CV_RGB(0, 255, 0),2,8);
            break;
        }
        default:
        {
            std::string buffer="fps:"+std::to_string(fps);
            cv::putText(bgr,buffer.c_str(),cvPoint(40,40),cv::FONT_HERSHEY_PLAIN,2,CV_RGB(0, 255, 0),2,8);
            break;
        }
    }
    video_mutex.unlock();
}

void cdk_result_model(void *result,void *param)
{
    DP_MODEL_NET model=*((DP_MODEL_NET*)param);
    switch (model)
    {
        case DP_AGE_NET:
        {
            sample_model.print_age_net_result(result);
            break;
        }
        case DP_GENDER_NET:
        {
            sample_model.print_gender_net_result(result);
            break;
        }
        case DP_ALEX_NET:
        {
            sample_model.print_alexnet_result(result);
            break;
        }
        case DP_GOOGLE_NET:
        {
            sample_model.print_googlenet_result(result);
            break;
        }
        case DP_RES_NET:
        {
            sample_model.print_resnet_result(result);
            break;
        }
        case DP_SQUEEZE_NET:
        {
            sample_model.print_squeezenet_result(result);
            break;
        }
        case DP_TINI_YOLO_NET:
        {
            sample_model.print_tiny_yolov1_net_result(result,box_demo,&num_box_demo);
            break;
        }
		case DP_TINY_YOLO_V2_FACE_NET:
		{
			std :: vector < float > probs;
			std :: vector <cv ::Rect> boxes;
			sample_model.print_tiny_yolov2_face_result(result, probs, boxes);
			int index=0;
			for (int i=0; i<boxes.size(); i++)
			{
				if(probs[i]>0)
				{
					box_demo[index].xmin=boxes[i].x;
        			box_demo[index].xmax=boxes[i].x+boxes[i].width;
        			if(box_demo[index].xmin>1280)
            			box_demo[index].xmax=1280;
        			box_demo[index].ymin=boxes[i].y;
        			box_demo[index].ymax=boxes[i].y+boxes[i].height;
        			if(box_demo[index].ymax>960)
            			box_demo[index].ymax=960;
					index++;
				}
			}
			cv::Mat tmpmean,stdtmp;
			num_box_demo=index;
			printf("\n num_box:%d\n",num_box_demo);
	    		dp_send_second_image_num(num_box_demo);
			if (num_box_demo>0)
			{
				cv::Mat tmpmean,stdtmp;
				for(int i=0;i<num_box_demo;i++)
				{
					cv::Rect SrcImgROI(box_demo[i].xmin,box_demo[i].ymin,box_demo[i].xmax-box_demo[i].xmin,box_demo[i].ymax-box_demo[i].ymin); 
					cv::Mat tmp=bgr(SrcImgROI);
					cv::Mat tmp_resized;
					cv::resize(tmp,tmp_resized,cv::Size(160,160));
					cv::Mat sample_coclr;
					cv::cvtColor(tmp_resized, sample_coclr, cv::COLOR_BGR2RGB);
					cv::meanStdDev(sample_coclr,tmpmean,stdtmp);
					printf("mean:%f,std:%f\n",tmpmean.at<double>(0,0),stdtmp.at<double>(0,0));
					double imgmean=tmpmean.at<double>(0,0);
					double imgstd=stdtmp.at<double>(0,0);
					float img_std=float(1/imgstd);
					printf("std_img:%f\n",img_std);
					cv::Mat sample_normalized;
                	sample_coclr.convertTo ( sample_normalized, CV_32FC3, img_std, -img_std*imgmean);
	            	for(int ii=0;ii<sample_normalized.rows;ii++)
	            	{
	            		for(int jj=0;jj<sample_normalized.cols;jj++)
	            		{
	            			imgbuffer[3*sample_normalized.cols*ii+jj*3+0]=sample_normalized.at<cv::Vec3f>(ii,jj)[0];
					imgbuffer[3*sample_normalized.cols*ii+jj*3+1]=sample_normalized.at<cv::Vec3f>(ii,jj)[1];
					imgbuffer[3*sample_normalized.cols*ii+jj*3+2]=sample_normalized.at<cv::Vec3f>(ii,jj)[2];
					//printf("sample_norm_channe:%f %f %f \n",imgbuffer[3*sample_normalized.cols*ii+jj*3],imgbuffer[3*sample_normalized.cols*ii+jj*3+1],imgbuffer[3*sample_normalized.cols*ii+jj*3+2]);
	            		}
	            	}
                	floattofp16((unsigned char *)tempbuffer, imgbuffer, 3*160*160); 
					dp_send_second_image(tempbuffer,2*160*160*3,1);
				}
			}
			break;
		}	
        case DP_SSD_MOBILI_NET:
        {
            	sample_model.print_ssd_mobilet_result(result,box_demo,&num_box_demo);
	    	printf("\n num_box:%d\n",num_box_demo);
	    	dp_send_second_image_num(num_box_demo);
	    	if(num_box_demo>0)
	    	{
			cv::Mat tmpmean,stdtmp;
			for(int i=0;i<num_box_demo;i++)
            		{  
            			cv::Rect SrcImgROI(box_demo[i].xmin,box_demo[i].ymin,box_demo[i].xmax-box_demo[i].xmin,box_demo[i].ymax-box_demo[i].ymin); 
                		cv::Mat tmp=bgr(SrcImgROI);
				cv::Mat tmp_resized;
				cv::resize(tmp,tmp_resized,cv::Size(160,160));
				cv::Mat sample_coclr;
				cv::cvtColor(tmp_resized, sample_coclr, cv::COLOR_BGR2RGB);
				cv::meanStdDev(sample_coclr,tmpmean,stdtmp);
				printf("mean:%f,std:%f\n",tmpmean.at<double>(0,0),stdtmp.at<double>(0,0));
				double imgmean=tmpmean.at<double>(0,0);
				double imgstd=stdtmp.at<double>(0,0);
				float img_std=float(1/imgstd);
				printf("std_img:%f\n",img_std);
				cv::Mat sample_normalized;
                		sample_coclr.convertTo ( sample_normalized, CV_32FC3, img_std, -img_std*imgmean);
                		std::vector<cv::Mat> sample_norm_channels;
	            		cv::split ( sample_normalized, sample_norm_channels );	//将sample_normalized各通道分离，满足caffe数据结构要求
                		for(int j=0; j < 160*160; j++ )
                		{
                			imgbuffer[3*j] =sample_norm_channels[0].data[j];
			     		imgbuffer[3*j + 1] = sample_norm_channels[1].data[j];
			        	imgbuffer[3*j + 2] = sample_norm_channels[2].data[j]; 
              			}
                		floattofp16((unsigned char *)tempbuffer, imgbuffer, 3*160*160); 
				dp_send_second_image(tempbuffer,2*160*160*3,1);
            		}
		}
            break;
        }
        case DP_INCEPTION_V1:
        {
            sample_model.print_inception_result(result);
            break;
        }
        case DP_MNIST_NET:
        {
            sample_model.print_mnist_net_result(result);
            break;
        }
        case DP_MOBILINERS_NET:
        {
            sample_model.print_mobilinet_net_result(result);
            break;
        }
        case DP_TINY_YOLO_V2_NET:
        {
            	sample_model.print_tiny_yolov2_result(result,box_demo,&num_box_demo);
		printf("\n num_box:%d\n",num_box_demo);
	    	dp_send_second_image_num(num_box_demo);
	    	if(num_box_demo>0)
	    	{
			cv::Mat tmpmean,stdtmp;
			for(int i=0;i<num_box_demo;i++)
            		{  
            			cv::Rect SrcImgROI(box_demo[i].xmin,box_demo[i].ymin,box_demo[i].xmax-box_demo[i].xmin,box_demo[i].ymax-box_demo[i].ymin); 
                		cv::Mat tmp=bgr(SrcImgROI);
				cv::Mat tmp_resized;
				cv::resize(tmp,tmp_resized,cv::Size(160,160));
				cv::Mat sample_coclr;
				cv::cvtColor(tmp_resized, sample_coclr, cv::COLOR_BGR2RGB);
				cv::meanStdDev(sample_coclr,tmpmean,stdtmp);
				printf("mean:%f,std:%f\n",tmpmean.at<double>(0,0),stdtmp.at<double>(0,0));
				double imgmean=tmpmean.at<double>(0,0);
				double imgstd=stdtmp.at<double>(0,0);
				float img_std=float(1/imgstd);
				printf("std_img:%f\n",img_std);
				cv::Mat sample_normalized;
                		sample_coclr.convertTo ( sample_normalized, CV_32FC3, img_std, -img_std*imgmean);
                		std::vector<cv::Mat> sample_norm_channels;
	            		cv::split ( sample_normalized, sample_norm_channels );	//将sample_normalized各通道分离，满足caffe数据结构要求
                		for(int j=0; j < 160*160; j++ )
                		{
                			imgbuffer[3*j] =sample_norm_channels[0].data[j];
			     		imgbuffer[3*j + 1] = sample_norm_channels[1].data[j];
			        	imgbuffer[3*j + 2] = sample_norm_channels[2].data[j]; 
              			}
                		floattofp16((unsigned char *)tempbuffer, imgbuffer, 3*160*160); 
				dp_send_second_image(tempbuffer,2*160*160*3,1);
            		}
		}
            	break;
        }
	case DP_FACE_NET:
	{
		sample_model.print_facenet_result(result,detector_people_name);
		break;
	}
    }

    return;
}

//双模型，解析获取box回传给板子
void cdk_two_result_model(void *result,void *param)
{
    DP_MODEL_NET model=*((DP_MODEL_NET*)param);
    switch (model)
    {
        case DP_SSD_MOBILI_NET:
        {
            sample_model.print_ssd_mobilet_result(result,box_demo,&num_box_demo);
            int img_width=1280;
            int img_height=960;
            int box_demo_num=0;
            if((num_box_demo<=2)&&(num_box_demo>0))
            {
                dp_image_box_t *box_second=(dp_image_box_t*)malloc(num_box_demo*sizeof(dp_image_box_t));

                for (int i = 0; i < num_box_demo; ++i)
                {
                    if(((box_demo[i].xmax-box_demo[i].xmin)!=0)&&((box_demo[i].ymax-box_demo[i].ymin)!=0))
                    {
                        box_second[box_demo_num].x1=box_demo[i].xmin;
                        box_second[box_demo_num].x2=box_demo[i].xmax;
                        if(box_second[box_demo_num].x2>img_width)
                            box_second[box_demo_num].x2=img_width;
                        box_second[box_demo_num].y1=box_demo[i].ymin;
                        box_second[box_demo_num].y2=box_demo[i].ymax;
                        if(box_second[box_demo_num].y2>img_height)
                            box_second[box_demo_num].y2=img_height;
                        box_demo_num++;
                    }
                }
                dp_send_first_box_image(box_demo_num, box_second);
                free(box_second);
            }
            else if(num_box_demo>2)
            {
                dp_image_box_t *box_second=(dp_image_box_t*)malloc(2*sizeof(dp_image_box_t));
                for(int i=0;i<num_box_demo;i++)
                {
                    if(((box_demo[i].xmax-box_demo[i].xmin)!=0)&&((box_demo[i].ymax-box_demo[i].ymin)!=0))
                    {
                        box_second[box_demo_num].x1=box_demo[i].xmin;
                        box_second[box_demo_num].x2=box_demo[i].xmax;
                        if(box_second[box_demo_num].x2>img_width)
                            box_second[box_demo_num].x2=img_width;
                        box_second[box_demo_num].y1=box_demo[i].ymin;
                        box_second[box_demo_num].y2=box_demo[i].ymax;
                        if(box_second[box_demo_num].y2>img_height)
                            box_second[box_demo_num].y2=img_height;
                        box_demo_num++;
                    }
                    if(box_demo_num==2)
                    {
                        break;
                    }
                }
                dp_send_first_box_image(2, box_second);
                free(box_second);
            }
            break;
        }
        case DP_TINY_YOLO_V2_NET:
        {
            sample_model.print_tiny_yolov2_result(result,box_demo,&num_box_demo);
            int img_width=1280;
            int img_height=960;
            int box_demo_num=0;
            if((num_box_demo<=2)&&(num_box_demo>0))
            {
                dp_image_box_t *box_second=(dp_image_box_t*)malloc(num_box_demo*sizeof(dp_image_box_t));

                for (int i = 0; i < num_box_demo; ++i)
                {
                    if(((box_demo[i].xmax-box_demo[i].xmin)!=0)&&((box_demo[i].ymax-box_demo[i].ymin)!=0))
                    {
                        box_second[box_demo_num].x1=box_demo[i].xmin;
                        box_second[box_demo_num].x2=box_demo[i].xmax;
                        if(box_second[box_demo_num].x2>img_width)
                            box_second[box_demo_num].x2=img_width;
                        box_second[box_demo_num].y1=box_demo[i].ymin;
                        box_second[box_demo_num].y2=box_demo[i].ymax;
                        if(box_second[box_demo_num].y2>img_height)
                            box_second[box_demo_num].y2=img_height;
                        box_demo_num++;
                    }
                }
                dp_send_first_box_image(box_demo_num, box_second);
                free(box_second);
            }
            else if(num_box_demo>2)
            {
                dp_image_box_t *box_second=(dp_image_box_t*)malloc(2*sizeof(dp_image_box_t));
                for(int i=0;i<num_box_demo;i++)
                {
                    if(((box_demo[i].xmax-box_demo[i].xmin)!=0)&&((box_demo[i].ymax-box_demo[i].ymin)!=0))
                    {
                        box_second[box_demo_num].x1=box_demo[i].xmin;
                        box_second[box_demo_num].x2=box_demo[i].xmax;
                        if(box_second[box_demo_num].x2>img_width)
                            box_second[box_demo_num].x2=img_width;
                        box_second[box_demo_num].y1=box_demo[i].ymin;
                        box_second[box_demo_num].y2=box_demo[i].ymax;
                        if(box_second[box_demo_num].y2>img_height)
                            box_second[box_demo_num].y2=img_height;
                        box_demo_num++;
                    }
                    if(box_demo_num==2)
                    {
                        break;
                    }
                }
                dp_send_first_box_image(2, box_second);
                free(box_second);
            }
            break;
        }
        default:
            break;
    }
}
