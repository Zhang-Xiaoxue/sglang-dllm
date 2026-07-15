export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# bash test.sh results/llada2_mini_gsm8k.csv mini gsm8k
# bash test.sh results/llada2_flash_gsm8k.csv flash gsm8k
bash test.sh results/llada2_flash_gpqa.csv flash gpqa
bash test.sh results/llada2_mini_gpqa.csv mini gpqa
