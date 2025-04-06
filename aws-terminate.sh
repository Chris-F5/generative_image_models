#!/bin/bash

set -e

aws ec2 terminate-instances --instance-ids $(cat .aws-id)
rm -rf aws-id aws-ip aws_instances
