from matplotlib import pyplot as plt
import json

def combined_graph(attention_data,gpn_data):
  
  parameters = ['energy','delay','network_lifetime','pdr','throughput','connectivity','routing_overhead', 'prize']
  specific_epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 99]

  for i in range (7):
    plt.plot(specific_epochs, attention_data[i], marker='o', label="Attention model") 
    plt.plot(specific_epochs, gpn_data[i],marker='o', label="GPN model") 
    
    plt.xlabel("No. of Epochs") 
    plt.ylabel("{}".format(parameters[i])) 
    plt.title("Comparison Graph for {}".format(parameters[i])) 
    
    plt.legend(loc="lower right") 
  
    plt.savefig("images/comparison/{}.jpg".format(parameters[i]))
    plt.show() 

def main():
    gpn_data = []
    attention_data = []

    with open("graph_data/attention.json", "r") as file:
        attention_data = json.load(file)

    with open("graph_data/gpn.json", "r") as file:
        gpn_data = json.load(file)

    combined_graph(attention_data, gpn_data)

if __name__ == "__main__":
    main()
