#! /bin/bash

# Install Virtualbox
sudo touch /etc/apt/sources.list.d/virtualbox.list
echo 'deb [arch=amd64] https://download.virtualbox.org/virtualbox/debian bionic contrib' | sudo tee -a /etc/apt/sources.list.d/virtualbox.list

wget -q https://www.virtualbox.org/download/oracle_vbox_2016.asc -O- | sudo apt-key add -
wget -q https://www.virtualbox.org/download/oracle_vbox.asc -O- | sudo apt-key add -

sudo apt-get update
echo virtualbox-ext-pack virtualbox-ext-pack/license select true | sudo debconf-set-selections
sudo apt-get install -y virtualbox virtualbox-ext-pack


# Delete any previously installed version of Docker
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install Docker
# sudo apt-get update
sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg-agent \
        software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
        "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) \
        stable"

sudo apt-get update
sudo apt-get install -y docker-ce=5:18.09.9~3-0~ubuntu-bionic  docker-ce-cli=5:18.09.9~3-0~ubuntu-bionic  containerd.io 

# Install kubectl
curl -LO https://storage.googleapis.com/kubernetes-release/release/v1.15.0/bin/linux/amd64/kubectl
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin/kubectl

# Install MiniKube and move to /usr/local/bin
curl -Lo minikube https://storage.googleapis.com/minikube/releases/v1.3.1/minikube-linux-amd64
chmod +x minikube
sudo cp minikube /usr/local/bin/
sudo rm minikube

# Start minikube
minikube start --cpus 4 --memory 12288 --disk-size=120g --extra-config=apiserver.authorization-mode=RBAC --extra-config=kubelet.resolv-conf=/run/systemd/resolve/resolv.conf --extra-config kubeadm.ignore-preflight-errors=SystemVerification

# Download and unpack the kfctl v1.0.1 tarball
wget https://github.com/kubeflow/kfctl/releases/download/v1.0.1/kfctl_v1.0.1-0-gf3edb9b_linux.tar.gz
tar -xvf kfctl_v1.0.1-0-gf3edb9b_linux.tar.gz && rm kfctl_v1.0.1-0-gf3edb9b_linux.tar.gz

sudo mv kfctl /usr/local/bin

# Set KF_NAME to the name of your Kubeflow deployment. This also becomes the
# name of the directory containing your configuration.
# For example, your deployment name can be 'my-kubeflow' or 'kf-test'.
echo
export KF_NAME=kf-demo
echo
# Set the path to the base directory where you want to store one or more
# Kubeflow deployments. For example, /opt/.
# Then set the Kubeflow application directory for this deployment.
export BASE_DIR=/home/$USER
echo
export KF_DIR=$BASE_DIR/$KF_NAME
echo $KF_DIR
echo
# Set the configuration file to use, such as the file specified below:
sudo mkdir -p ${KF_DIR}
sudo wget -P ${KF_DIR} "https://raw.githubusercontent.com/kubeflow/manifests/v1.0-branch/kfdef/kfctl_k8s_istio.v1.0.1.yaml"

# Generate and deploy Kubeflow:
cd ${KF_DIR} && sudo kfctl apply -V -f kfctl_k8s_istio.v1.0.1.yaml

# Check to ensure all pods are in running status.
sudo kubectl get pod -n kubeflow

# Check your setting of istio-ingressgateway service
export INGRESS_HOST=$(minikube ip)
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].nodePort}')

echo
echo Navigate to http://$INGRESS_HOST:$INGRESS_PORT to see your Kubeflow dashboard!
