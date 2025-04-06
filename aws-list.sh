#!/bin/bash

set -e

rm -f .aws_instances .aws-ip .aws-id

aws ec2 describe-instances \
    --filters "Name=image-id,Values=ami-061854b98d50d6dae" \
    --query "Reservations[*].Instances[*].[InstanceId,PublicIpAddress,State.Name]" \
    | tee .aws_instances

if [[ $(grep -e running -e pending .aws_instances | wc -l) -eq 0 ]] ; then
    echo "NO INSTANCES"
    exit 1
fi

if [[ $(grep -e running -e pending .aws_instances | wc -l) -ne 1 ]] ; then
    echo "UNEXPECTED INSTANCES"
    exit 1
fi

grep -e running -e pending .aws_instances | awk '{print $2}' > .aws-ip
grep -e running -e pending .aws_instances | awk '{print $1}' > .aws-id
