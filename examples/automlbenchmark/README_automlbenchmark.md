This README provides instructions to run AutoGluon on AutoML Benchmark

# Note

These instructions are fairly complicated due to nuances in how AutoMLBenchmark works 
and its lack of native support for benchmarking forks/branches of AutoML Frameworks.

If you only care about benchmarking official releases of AutoGluon with other AutoML Frameworks on the original 39 datasets, 
then you are best served referring to the official AutoMLBenchmark readme: https://github.com/openml/automlbenchmark

If you care about benchmarking a modified version of AutoGluon (or any AutoML framework), such as an in-progress PR, 
then the steps below are NECESSARY.

## Note on running the full benchmark

Please be aware that **the full AutoMLBenchmark is extremely resource intensive**. 
It consists of 1040 datasets (104 datasets with 10-fold CV), each of which are trained for 5 hours per framework.
This means each framework requires 5200 hours of compute for a single benchmark. 
Assuming a standard AWS account that can run 10 EC2 instances at once, **this would take 9 days to finish, costing $1996.80 in compute** (using On-Demand AWS pricing).

While we plan to make a light-weight version of the benchmark for users to experiment with, this is not yet finished.

### Ways to cut down on price:

#### 1. Use `-f 0` in the `runbenchmark.py` command.

This will cut down on compute by 10x by only doing the first fold in the 10-fold CV instead of all 10.

5200 hrs -> 520 hrs

#### 2. Only run 1 hour benchmarks, ignore 4 hour benchmarks.

This will cut down on compute by 5x by only training for 1 hour instead of 5.

520 hrs -> 104 hrs

# Instructions

Option 1: Copy existing fork

1. Fork/Copy an existing working branch and edit slightly to make it work.
    1. Downside: May be out-of-date with automlbenchmark mainline or have other changes you don't want.
    2. Example forked branch: https://github.com/Innixma/automlbenchmark/tree/ag-2021_12_21
    3. If using this method, still refer to Option 2 Part 7+ to ensure the files are correct. You will still need to make code edits.

Option 2: From scratch

1. Fork automlbenchmark: https://github.com/openml/automlbenchmark
   1. Note: Only required if you want to do more than benchmark stable release AutoGluon locally. 
2. Create venv:
```
VENV=automlbenchmark
mkdir -p ~/virtual
python3 -m venv ~/virtual/$VENV
source ~/virtual/$VENV/bin/activate
pip install -U pip
```
3. Clone to local machine:
```
# Your github account name, i.e.: innixma
GIT_USER=innixma
REPO=https://github.com/$GIT_USER/automlbenchmark
git clone $REPO
```
4. Install:
```
# Tested with Python 3.6 and 3.7, MacOS and Linux, unknown if other versions work
PROJECT_PATH="$(pwd)/automlbenchmark"
pip install -r $PROJECT_PATH/requirements.txt
```
5. (Optional) Load project into IDE such as PyCharm
6. Test AutoGluon. This will create an AG venv, install AG, and run AG on a few datasets, will take a few minutes.
```
# This runs the latest stable release of AutoGluon.
mkdir -p benchmark_tmp
cd benchmark_tmp
python $PROJECT_PATH/runbenchmark.py AutoGluon test
# For presets='best_quality':
# python $PROJECT_PATH/runbenchmark.py AutoGluon_bestquality test
```
7. To run a forked branch of AutoGluon
    1. Make a new branch in automlbenchmark
    2. Edit `automlbenchmark/frameworks/AutoGluon/setup.sh` and add the following code after the `else`
    ```
    if [[ "$VERSION" == "stable" ]]; then
        PIP install --no-cache-dir -U ${PKG}
    elif [[ "$VERSION" =~ ^[0-9] ]]; then
        PIP install --no-cache-dir -U ${PKG}==${VERSION}
    else
        # Your new code
        # Your branch name
        VERSION="MY_AUTOGLUON_BRANCH_NAME"  
        # replace with the fork of autogluon your branch lives in.
        # if in awslabs (main) repo, this line is not necessary
        REPO="https://github.com/MY_GITHUB_ACCOUNT/autogluon.git"
    ```
    3. You may need to manually delete the AG venv from previous runs, or specify `-s force` in benchmark command so old venv is overwritten.
    4. Run benchmark (The AG branch MUST be publicly available on github):
    ```
    # Ensure :latest is added so it uses custom branch instead of pip install.
    python $PROJECT_PATH/runbenchmark.py AutoGluon:latest test -s force
    ```
    
8. Advanced: Run a distributed benchmark test of AutoGluon on AWS
   1. First ensure you have a fork and new branch of automlbenchmark, various changes must be made to this branch.
   2. Edit `automlbenchmark/resources/config.yaml`, changing the following lines and keep the rest untouched:
   
     ```yaml
   
     ---
     project_repository: https://github.com/MY_GIT_USER/automlbenchmark#MY_AUTOMLBENCHMARK_BRANCH_NAME
     max_parallel_jobs: 2000  # 2000 so we effectively don't have a limit on parallel EC2 instances
     benchmarks:
       overhead_time_seconds: 72000  # 72000 so we don't randomly stop instances if they take a bit longer than specified
     aws:
       region: 'us-east-1'  # us-east-1 or whatever region you plan to launch instances
       s3:
         bucket: YOUR_S3_BUCKET_NAME  # make the bucket in S3 first, specify a new one to isolate runs from other users (requires creation)
         root_key: ec2/2021_12_21/  # subdirectory in bucket where results are saved, try to keep in sync with what you are testing
                                    # avoid re-using between multiple runs as it is easy to confuse which results are from what experiment
       ec2:
         volume_type: gp2  # standard is very slow, prefer gp2
       max_timeout_seconds: 72000  # just to avoid any strange timeouts
       overhead_time_seconds: 28800  # just to avoid any strange timeouts
     ```
   3. Edit `automlbenchmark/resources/constraints.yaml`, changing the `min_vol_size_mb` to `100000` in all cases.
      - This allows 100 GB of space for each instance, enough to avoid OOD errors.
   4. Do the edits in part 7 if they weren't already done.
   5. Commit and push the change so it is available on github.
   6. Create an EC2 instance that will act as the main node that launches and keeps track of all other instances.
      - Ensure this is at least an m5.2xlarge with 500 GB of disk in main partition.
      - Ensure this EC2 instance has the ability to access S3 and create/delete other EC2 instances. (EC2 Admin level permissions)
      - This instance will not train models, but Deep Learning AMI worked fine for me.
   7. Git clone your forked branch of automlbenchmark to the EC2 instance and make a virtual environment + install (Redo steps 2-4)
   8. Test an AWS run:
   ```
   mkdir -p benchmark_tmp
   cd benchmark_tmp
   # Run test suite on AWS, with at most 10 parallel instances at once, on AutoGluon stable release
   python $PROJECT_PATH/runbenchmark.py AutoGluon_bestquality test -m aws -p 10
   # Run with your custom AutoGluon branch:
   # python $PROJECT_PATH/runbenchmark.py AutoGluon_bestquality:latest test -m aws -p 10
   ```
   Once started, you should see the instances start up on EC2. Results will be saved to S3 in your `$bucket/$root_key` location specified in `config.yaml`.
   
9. Advanced: Run in parallel the full AutoML Benchmark with custom AutoGluon branch (4160 machines, 10400 hrs of m5.2xlarge compute)
   1. Copy `ag.yaml` into your automlbenchmark branch: https://github.com/Innixma/automlbenchmark/blob/ag-2021_12_21/resources/benchmarks/ag.yaml
   2. Commit and Push
   3. On EC2: fetch and pull
   4. On EC2:
   ```
   mkdir -p benchmark_tmp
   cd benchmark_tmp
   # Run full benchmark suite (1h8c) on AWS, fully parallel, on AutoGluon custom branch / master with 'best_quality' preset
   # DO NOT run all 4 of these at the same time, preferably do them sequentially.
   # Consider adding -f 0 to reduce compute by 10x (only train 1 fold instead of all 10), lowers confidence in results, but cheaper and faster.
   # Results should be available within 3-4 hours for 1h8c, and within 7-8 hours for 4h8c.
   # If you want to run on stable release of AutoGluon, remove :latest
   # python $PROJECT_PATH/runbenchmark.py AutoGluon_bestquality:latest ag 1h8c -m aws -p 1500
   # with 4h8c
   # python $PROJECT_PATH/runbenchmark.py AutoGluon_bestquality:latest ag 4h8c -m aws -p 1500
   # without best_quality preset
   # python $PROJECT_PATH/runbenchmark.py AutoGluon:latest ag 1h8c -m aws -p 1500
   # python $PROJECT_PATH/runbenchmark.py AutoGluon:latest ag 4h8c -m aws -p 1500
   ```
   
10. Advanced: Run distributed benchmark on your custom datasets on AWS

    1. Prepare your datasets in s3 bucket with the following prefix: `arn:aws:s3:::automl-benchmark-*/ec2/*`. If you plan to have multiple folds, put them into separate files.

    2. Create your custom benchmark yaml file under `automlbenchmark/resources/`, for example `custom_test.yaml`

    3. Example benchmark yaml file:

        ```yaml
        - name: custom_dataset1
          dataset:
            train: 
              - 's3://automl-benchmark-xxx/ec2/custom_datasets/train0.csv'
              - 's3://automl-benchmark-xxx/ec2/custom_datasets/train1.csv'
              - 's3://automl-benchmark-xxx/ec2/custom_datasets/train2.csv'
            test: 
              - 's3://automl-benchmark-xxx/ec2/custom_datasets/test0.csv'
              - 's3://automl-benchmark-xxx/ec2/custom_datasets/test1.csv'
              - 's3://automl-benchmark-xxx/ec2/custom_datasets/test2.csv'
            target: class # This is the label to be predicted
          folds: 3
        
        ```

    4. Follow previous guide for running distributed benchmark on AWS. Remember to push and commit your changes.

    5. To run the benchmark:

       ```
       mkdir -p benchmark_tmp
       cd benchmark_tmp
       # Run full benchmark suite (1h8c) on AWS, fully parallel, on AutoGluon custom branch / master with 'best_quality' preset
       # Consider adding -f 0 to reduce compute by 10x (only train 1 fold instead of all 10), lowers confidence in results, but cheaper and faster.
       # If you want to run on stable release of AutoGluon, remove :latest
       # -p 1500 is just an example, change it to whatever suites your own benchmark
       # python $PROJECT_PATH/runbenchmark.py AutoGluon_bestquality:latest custom_test 1h8c -m aws -p 1500
       ```

11. Aggregate AWS S3 results.

     1. WIP

12. Analyze aggregated results.
     1. WIP