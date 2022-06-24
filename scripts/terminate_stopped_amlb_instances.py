
import argparse
import time

import boto3


def terminate_stopped_ec2_amlb(region='us-east-1', dry_run=True):
    """
    Terminates all stopped EC2 instances from automlbenchmark in a given region.
    This is used to cleanup the AWS account to avoid running into errors in future runs when many stopped instances exist.
    Only run this if you know what you are doing. Set `dry_run=False` to actually terminate the instances.
    """
    ec2 = boto3.resource('ec2', region)
    name_prefix = 'benchmark_aws.'

    total_instance_count = 0
    stopped_instance_count = 0
    instances_to_terminate = []
    for instance in ec2.instances.all():
        total_instance_count += 1
        if instance.state['Name'] == 'stopped':
            stopped_instance_count += 1
            for tag in instance.tags:
                if tag['Key'] == 'Name':
                    name = tag['Value']
                    if name.startswith(name_prefix):
                        print(f'Planning to terminate: {name}')
                        instances_to_terminate.append(instance)
                        break

    terminating_instance_count = len(instances_to_terminate)
    print(f'Total Instances        : {total_instance_count}')
    print(f'Stopped Instances      : {stopped_instance_count}')
    print(f'Stopped AMLB Instances : {len(instances_to_terminate)}')
    print(f'Terminating {terminating_instance_count} stopped AMLB instances in 20 seconds (kill process to abort!)')
    if dry_run:
        print('DRY RUN... INSTANCES WILL NOT BE TERMINATED')
    time.sleep(20)
    if dry_run:
        print(f'Skipping termination because `dry_run==True`.')
        return
    print(f'Terminating...')
    for i, instance in enumerate(instances_to_terminate):
        instance.terminate()
        print(f'Terminated {i + 1}/{terminating_instance_count} instances...')
    print('Finished terminating instances')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--region', type=str, help="Region to delete from", default='us-east-1', nargs='?')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--no-dry_run', dest='dry_run', action='store_false')
    parser.set_defaults(dry_run=True)

    args = parser.parse_args()

    terminate_stopped_ec2_amlb(
        region=args.region,
        dry_run=args.dry_run,
    )
    # Uncomment below line to terminate instances
    # terminate_stopped_ec2_amlb(dry_run=False)
