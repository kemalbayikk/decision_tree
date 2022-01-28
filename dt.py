# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 17:35:35 2021

@author: Kemal
"""

#Kodu yazarken X ve y şeklinde dataseti ayırmadım hepsini X ' e parametre olarak gönderiyorum ve fonksiyonların içinde ayrımı yapıyorum

class DecisionTreeClassifier:
    
    def __init__(self, max_depth: int):
        global depth
        depth = max_depth
        pass
    
    def fit(self, X, y):
        global tree
        global columns
        columns = X[0]
        tree = self.decision_tree_algorithm(X,max_depth=depth)
        return tree

    def predict(self, X):
        return self.calculate_accuracy(X, tree)
        
    #Listteki unique verilen bulunması
    def unique(self,list1):
 
        unique_list = []
        for x in list1:
            if x not in unique_list:
                unique_list.append(x)
        return unique_list
    
    
    #Tüm column değerlerini unique olacak şekilde ayıran fonksiyon
    def get_splits(self,data):
        splits = {}
        values = []
        n_columns = len(data[0])
        n_rows = len(data)
        for column_index in range(n_columns - 1):
            splits[column_index] = []
            for k in range (n_rows):
                values.append(data[k][column_index])
                unique = sorted(self.unique(values))
            for index in range(len(unique)):
                if index != 0:
                    current = unique[index]
                    previous = unique[index - 1]
                    split = (current + previous) / 2
                
                    splits[column_index].append(split)
        return splits
    
    
    #Belirlenen value'ya göre left right olarak ayıran fonksiyon
    def split_data(self,data, split_column, split_value):
        data_temp_left = []
        data_temp_right = []
        data_left = []
        data_right = []
        splitted_column_values = []
        for k in range (len(data)):
            splitted_column_values.append(data[k][split_column])
        
        for k in range (len(splitted_column_values)):
            if splitted_column_values[k] <= split_value:
                for i in range (len(data[0])):
                    data_temp_left.append(data[k][i])
                    
                data_left.append(data_temp_left)   
                data_temp_left = []
                
            else: 
                 for i in range (len(data[0])):
                    data_temp_right.append(data[k][i])
                    
                 
                 data_right.append(data_temp_right)
                 data_temp_right = []
        
        return data_left,data_right 
    
    #Her species'den kaç tane olduğu
    def class_counts(self, rows):
        counts = {}  
        for row in rows:
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts
    
    #Gini impurity
    def gini(self,rows):

        counts = self.class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl**2
        return impurity
    
    #Gini impurity'e göre information gain hesabı
    def info_gain(self,left, right, current_uncertainty):

        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * self.gini(left) - (1 - p) * self.gini(right)
    
    #Datayı hangi columna ve hangi value'ya göre böleceğimizin information gain ile hesaplanması
    def split_gini(self,data, splits):
        best_gain = 0  
        current_impurity = self.gini(data)
        for column_index in splits:
            for value in splits[column_index]:
                data_left, data_right = self.split_data(data, split_column = column_index, split_value = value)
                gain = self.info_gain(data_left, data_right, current_impurity)
            
                if gain >= best_gain:
                    best_gain = gain
                    best_split_column = column_index
                    best_split_value = value
        return best_split_column, best_split_value
    
    #Datada farklı speciesler var mı yoksa data pure mu?
    def check_purity(self,data):
        n_rows = len(data)
        
        values = []
        for k in range (n_rows):
            values.append(data[k][len(data[0]) - 1])
        unique_values = sorted(self.unique(values))
        
        
    
        if len(unique_values) == 1:
            return True
        else:
            return False
     
    #Verilen datada en çok hangi species varsa onu seç    
    def classify_data(self,data,all_data):
    
        n_rows = len(data)
        max_count = 0
        count_index = 0
        
        values = []
        counts_unique = []
        species = []
        unique_species = []
        
        for k in range (len(all_data)):
            species.append(all_data[k][len(all_data[0]) - 1 ])
        
        unique_species = sorted(self.unique(species))
    
        if len(data) == 0:
            classification = unique_species[0]
        else:
            for k in range (n_rows):
                values.append(data[k][len(data[0]) - 1])
        
            species = values
            unique_values = sorted(self.unique(values))
            counts_unique_values = self.class_counts(data)
            if unique_species[0] in counts_unique_values :
                counts_unique.append(counts_unique_values[unique_species[0]])
            if len(unique_species)  > 1:
                if unique_species[1] in counts_unique_values :
                    counts_unique.append(counts_unique_values[unique_species[1]])
            if len(unique_species ) > 2:
                if unique_species[2] in counts_unique_values :
                    counts_unique.append(counts_unique_values[unique_species[2]])
        
            for k in range (len(counts_unique)): 
                if counts_unique[k] > max_count:
                    max_count = counts_unique[k]
                    count_index = k
            classification = unique_values[count_index]
        
        return classification
    
    #Decision Tree Algoritması
    def decision_tree_algorithm(self,df_list, max_depth, counter=0):
        data = []
        if counter == 0:
            global COLUMN_HEADERS
            COLUMN_HEADERS = df_list[0]
            for i in range (1, len(df_list)):
                data.append(df_list[i])
        else:
            for i in range (1, len(df_list)):
                data.append(df_list[i])
        
    
        if (self.check_purity(data)) or (len(data) < 2) or (counter == max_depth):
            classification = self.classify_data(data,df_list)
            
            return classification
    
        
        else:    
            counter += 1
    
            splits = self.get_splits(data)
            split_column, split_value = self.split_gini(data, splits)
            data_below, data_above = self.split_data(data, split_column, split_value)
            
            feature_name = COLUMN_HEADERS[split_column]
            question = "{} <= {}".format(feature_name, split_value)
            sub_tree = {question: []}
            
            yes_answer = self.decision_tree_algorithm(data_below, counter, max_depth)
            no_answer = self.decision_tree_algorithm(data_above, counter, max_depth)
            if yes_answer == no_answer:
                sub_tree = yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)
            
            return sub_tree
    
    #Oluşan decision tree'ye göre dataların classify edilmesi    
    def classify(self,row, tree):
        dictlist = []
        column_names = columns
        feature_index = 0
        for key, value in tree.items():
            temp = [key,value]
            dictlist.append(temp)
        
        question = str(dictlist[0][0])
        feature_name, comparison_operator, value = question.split()
        for k in range (len(column_names)):
            if column_names[k] == feature_name:
                feature_index = k      

        if row[feature_index] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]

        if not isinstance(answer, dict):
            return answer
        
        else:
            residual_tree = answer
            return self.classify(row, residual_tree)
    
        
    #Test setine göre Accuracy Presicion Recall F1-Score Hesaplanması
    def calculate_accuracy(self,df_list, tree):
        classification = []
        correct_classification = 0
        species = []
        example = []
        confusion_matrix = [[0,0,0],[0,0,0],[0,0,0]]
        presicions = []
        recalls = []
        unique_species = []
        f1s = []
        for k in range (len(df_list) ):
            for j in range (len(df_list[0])):
                example.append(df_list[k][j])
            classification.append(self.classify(example, tree))
            example = []
    
            
        
        for k in range (len(df_list)):
            species.append(df_list[k][len(df_list[0]) - 1 ])
        
        unique_species = sorted(self.unique(species))
    
        for k in range (len(species)):
            if species[k] == unique_species[0] and classification[k] == unique_species[0]:
                correct_classification+=1
                confusion_matrix[0][0]+=1
            if species[k] == unique_species[0] and classification[k] == unique_species[2]:
                confusion_matrix[0][1]+=1
            if species[k] == unique_species[0] and classification[k] == unique_species[1]:
                confusion_matrix[0][2]+=1
                
            if species[k] == unique_species[2] and classification[k] == unique_species[2]:
                correct_classification+=1
                confusion_matrix[1][1]+=1
            if species[k] == unique_species[2] and classification[k] == unique_species[0]:
                confusion_matrix[1][0]+=1
            if species[k] == unique_species[2] and classification[k] == unique_species[1]:
                confusion_matrix[1][2]+=1
                
            if species[k] == unique_species[1] and classification[k] == unique_species[1]:
                correct_classification+=1
                confusion_matrix[2][2]+=1
            if species[k] == unique_species[1] and classification[k] == unique_species[2]:
                confusion_matrix[2][1]+=1
            if species[k] == unique_species[1] and classification[k] == unique_species[0]:
                confusion_matrix[2][0]+=1
                
        print("Confusion Matrix : ",confusion_matrix)  
        presicions.append(confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[0][2]))
        presicions.append(confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0] + confusion_matrix[1][2])) 
        presicions.append( confusion_matrix[2][2] / (confusion_matrix[2][2] + confusion_matrix[2][1] + confusion_matrix[2][0]))
        
        recalls.append(confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0] + confusion_matrix[2][0]))
        recalls.append(confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1] + confusion_matrix[2][1])) 
        recalls.append(confusion_matrix[2][2] / (confusion_matrix[2][2] + confusion_matrix[0][2] + confusion_matrix[1][2])) 
        
        
        f1s.append(2*((presicions[0]*recalls[0]) / (presicions[0]+recalls[0])) )
        f1s.append(2*((presicions[1]*recalls[1]) / (presicions[1]+recalls[1])) )
        f1s.append(2*((presicions[2]*recalls[2]) / (presicions[2]+recalls[2])) )
        print("Iris-setosa presicion : ",presicions[0])
        print("Iris-virginica presicion : ",presicions[1])
        print("Iris-versicolor presicion : ",presicions[2])
        
        print("Iris-setosa recall : ",recalls[0])
        print("Iris-virginica recall : ",recalls[1])
        print("Iris-versicolor recall : ",recalls[2])
        
        print("Iris-setosa F1-Score : ",f1s[0])
        print("Iris-virginica F1-Score : ",f1s[1])
        print("Iris-versicolor F1-Score : ",f1s[2])
           
                
        accuracy = correct_classification / len(species) * 100
        print("Accuracy : ",accuracy)
        return species,classification
    

