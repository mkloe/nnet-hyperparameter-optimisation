#directory
setwd("/Users/marcinzurek/Desktop/Studia/Magisterka/Data Mining/projekt")

#wczytanie danych
data <- read.csv("zbior.csv")

#podstawowe informacje o danych
data
head(data)
summary(data)

#liczba NA według kolumn
colSums(is.na(data))

##############################
########SIEC NEURONOWA########
##############################

head(data)

#wybór zmiennych dla sieci neuronowej
variables <- colnames(data)
variables
nnetVariables <- variables[2:27]
data <- data[nnetVariables]

#faktoryzacja zmiennej celu
data$flaga <- as.factor(data$flaga)

#wybor zbioru treningowego i walidacyjnego
uczaceIndeksy <- sample.int(nrow(data), 0.75*nrow(data), replace = FALSE)
zbioruczacy <- data[uczaceIndeksy,]
walidacyjnyIndeksy <- setdiff(1:nrow(data), uczaceIndeksy)
zbiorwalidacyjny <- data[walidacyjnyIndeksy,]


#pakiety
#install.packages(c("nnet", "caret", "e1071"))
library(caret)
library(nnet)
library(e1071)


#normalizacja zmiennych ciągłych
skalowany_uczacy <- zbioruczacy
skalowany_walidacyjny <- zbiorwalidacyjny

for (i in 1:3){
  wektor <- scale(skalowany_uczacy[,i])
  centerVals <- attr(wektor, 'scaled:center')
  scaleVals <- attr(wektor, 'scaled:scale')
  skalowany_uczacy[,i] <- wektor
  skalowany_walidacyjny[,i] <- scale(skalowany_walidacyjny[,i],
                                     center=centerVals,scale=scaleVals)
}

skalowany_uczacy
skalowany_walidacyjny

##############################################
########symulacja - siec neuronowa############
##############################################
data_train <- skalowany_uczacy[1:500,] 
data_train
data_test <- skalowany_walidacyjny[1:500,]
data_test


#funkcja na sprawdzenie czasu symulacji
comparennet_time <- function(dane_train, 
                             sim_steps = 30,  #liczba iteracji symulacji. By default = 30 poniewaz jest to symulacja stochastyczna.
                             layer_size_min = 10,  #minimalna liczba neuronow w warstwie ukrytej
                             layer_size_max = 30,  #maksymalna liczba neuronow w warstwie ukrytej
                             layer_size_step = 5,  #krok iteracji dla petli for - tworzy wektor seq(layer_size_min, layer_size_max, by=layer_size_step)
                             skip_vec = c(TRUE, FALSE), #wektor paramterow skip dla ktorych ma zostac wykonana symulacja. Skip - polaczenia pomiedzy warstwa wejsciowa i wyjsciowa.
                             decay_min = 0,  #minimalna wartosc parametru decay
                             decay_max = 0.1, #maksymalna wartosc parametru decay
                             decay_step = 0.02){ #krok iteracji dla petli for - tworzy wektor seq(decay_min, decay_max, by=decay_step)
  
  size_vector <- seq(layer_size_min, layer_size_max, by=layer_size_step)
  size_vector_length <- length(size_vector)
  decay_vector <- seq(decay_min, decay_max, by=decay_step)
  decay_vector_length <- length(decay_vector)
  skip_vec_length <- length(skip_vec)
  
  #czas symulacji
  start <- Sys.time()
  nnet(flaga~., dane_train, size=(layer_size_max+layer_size_min)/2, decay=0.05,
       maxit=100, MaxNWts=5000, trace=F, skip=T)
  end <- Sys.time()
  tot_time_min <- as.numeric((end-start))*decay_vector_length*skip_vec_length*sim_steps*size_vector_length/60
  cat("Oczekiwany czas symulacji w minutach:", tot_time_min, "\n")
}

#sprawdzenie przykladowych czasow symulacji
comparennet_time(data_train)
comparennet_time(skalowany_uczacy)


#funkcja na wykonanie symulacji
comparennet <- function(dane_train,
                        dane_test, 
                        sim_steps = 30,  #liczba iteracji symulacji. By default = 30 poniewaz jest to symulacja stochastyczna.
                        layer_size_min = 10,  #minimalna liczba neuronow w warstwie ukrytej
                        layer_size_max = 30,  #maksymalna liczba neuronow w warstwie ukrytej
                        layer_size_step = 5,  #krok iteracji dla petli for - tworzy wektor seq(layer_size_min, layer_size_max, by=layer_size_step)
                        skip_vec = c(TRUE, FALSE), #wektor paramterow skip dla ktorych ma zostac wykonana symulacja. Skip - polaczenia pomiedzy warstwa wejsciowa i wyjsciowa.
                        decay_min = 0,  #minimalna wartosc parametru decay
                        decay_max = 0.1, #maksymalna wartosc parametru decay
                        decay_step = 0.02){ #krok iteracji dla petli for - tworzy wektor seq(decay_min, decay_max, by=decay_step)
  
  #inicjalizacja zmiennych
  size_vector <- seq(layer_size_min, layer_size_max, by=layer_size_step)
  size_vector_length <- length(size_vector)
  decay_vector <- seq(decay_min, decay_max, by=decay_step)
  decay_vector_length <- length(decay_vector)
  skip_vec_length <- length(skip_vec)
  
  #czas symulacji
  start <- Sys.time()
  nnet(flaga~., dane_train, size=(layer_size_max+layer_size_min)/2, decay=0.05,
       maxit=100, MaxNWts=5000, trace=F, skip=T)
  end <- Sys.time()
  tot_time_min <- as.numeric((end-start))*decay_vector_length*skip_vec_length*sim_steps*size_vector_length/60
  cat("Oczekiwany czas symulacji w minutach:", tot_time_min, "\n")
  
  #symulacja
  i <- 0
  wyniki <-  matrix(0, nrow = decay_vector_length *skip_vec_length*sim_steps*size_vector_length, ncol = 9)
  for (skip in skip_vec) {
    
    if (skip == TRUE){skip1 = 1} else {skip1 = 0}
    
    for (decay in decay_vector) {
      
      for (size in size_vector){
      
        set.seed(0)
        for (j in 1:sim_steps){
          i <- i+1
          
          #informacje o stanie iteracji
          cat("Iteracja ", i, "z ", decay_vector_length*skip_vec_length*sim_steps*size_vector_length, "\n")
          time_passed_min <- (as.numeric(Sys.time()) - as.numeric(start))/60
          time_left <- tot_time_min - time_passed_min
          cat("Czas symulacji w minutach: ", time_passed_min, "\n")
          cat("Pozostaly czas symulacji w minutach: ", time_left, "\n")
          
          #uczenie sieci i klasyfikacja
          siec_ucz_Base <- nnet(flaga~., dane_train, size=size, decay=decay,
                                maxit=100, MaxNWts=5000, trace=F, skip = skip)
          
          klasyfikacja_ucz <- predict(siec_ucz_Base,dane_train,type="class")
          F1_train <- caret::confusionMatrix(as.factor(klasyfikacja_ucz), 
                                             as.factor(dane_train$flaga))[["byClass"]][["F1"]]
          Accuracy_train <- caret::confusionMatrix(as.factor(klasyfikacja_ucz), 
                                                   as.factor(dane_train$flaga))[["overall"]][["Accuracy"]]
          
          klasyfikacja_walid <- predict(siec_ucz_Base, dane_test,type="class")  
          F1_test <- caret::confusionMatrix(as.factor(klasyfikacja_walid), 
                                            as.factor(dane_test$flaga))[["byClass"]][["F1"]]
          Accuracy_test <- caret::confusionMatrix(as.factor(klasyfikacja_walid), 
                                                  as.factor(dane_test$flaga))[["overall"]][["Accuracy"]]
          
          wiersz <- c(skip1, decay, size, j, siec_ucz_Base$value, F1_train, Accuracy_train, F1_test, Accuracy_test)
          wyniki[i,] <- wiersz
        }
      }
    }
    
  }
  
  columnnames <- c("Skip", "Decay", "Size", "Sim_step", "LossValue", "F1_train","Accuracy_train", "F1_test", "Accuracy_test")
  a <- data.frame(wyniki)
  colnames(a) <- columnnames
  return(a[order(-a$F1_test),])
}

#sprawdzenie dzialania funkcji na malym zbiorze
results <- comparennet(data_train, data_test, sim_steps = 1)
results
library(dplyr)
results %>% 
  group_by(Skip, Decay, Size) %>% 
  summarise(LossValue = mean(LossValue), 
            F1_train = mean(F1_train),
            Accuracy_train =mean(Accuracy_train),
            F1_test = mean(F1_test),
            Accuracy_test =mean(Accuracy_test)) %>% 
  arrange(desc(F1_test))


#wykonanie funkcji na pelnych zbiorach
final_results_acc <- comparennet(skalowany_uczacy, skalowany_walidacyjny)
final_results_acc
library(dplyr)
final_results_acc_sum <- final_results_acc %>% 
  group_by(Skip, Decay, Size) %>% 
  summarise(LossValue = mean(LossValue), 
            F1_train = mean(F1_train),
            Accuracy_train =mean(Accuracy_train),
            F1_test = mean(F1_test),
            Accuracy_test =mean(Accuracy_test)) %>% 
  arrange(desc(Accuracy_test))

#wyswietlenie pierwszych 10 rekordow o najwyzszym wskazniku accuracy_test
View(head(final_results_acc_sum, 10))

##########################################
#######analiza najlepszego zbioru#########
##########################################

siec_ucz <- nnet(flaga~., skalowany_uczacy, size=30, decay=0.06,
                      maxit=300, MaxNWts=5000, trace=T, skip = T)

siec_ucz
#predykcja na zbiorze uczacym
klasyfikacja_ucz<-predict(siec_ucz, skalowany_uczacy, type="class")
klasyfikacja_ucz

skalowany_uczacy$wynik_klasyfikacji <- klasyfikacja_ucz

#macierz pomylek - zbior treningowy
confusion_train <- caret::confusionMatrix(as.factor(klasyfikacja_ucz), as.factor(skalowany_uczacy$flaga))
confusion_train
#Klasyfikacja na zbiorze walidacyjnym
klasyfikacja_walid <- predict(siec_ucz, skalowany_walidacyjny, type="class")
klasyfikacja_walid

# dodanie wektora z wynikami klasyfikacji do zbioru walidacyjnego
skalowany_walidacyjny$wynik_klasyfikacji<-klasyfikacja_walid

#macierz pomylek - zbior testowy
confusion_test <- caret::confusionMatrix(as.factor(klasyfikacja_walid), as.factor(skalowany_walidacyjny$flaga))
confusion_test

summary(skalowany_walidacyjny$flaga)

##Krzywa ROC
library(pROC)
library(ggplot2)

klasyfikacja_roc <- predict(siec_ucz, skalowany_walidacyjny, type="raw")
roc_curve <- roc(response = skalowany_walidacyjny$flaga, predictor = as.numeric(klasyfikacja_roc), 
                 direction = "<", levels = c(0,1))
roc_curve

ggroc(roc_curve, legacy.axes = T) +
  geom_abline(slope = 1 ,intercept = 0) + 
  ggtitle("Krzywa ROC") +
  theme_bw() +
  xlab('100% - Specificity') +
  ylab('100% - Sensitivity') +
  scale_x_continuous(breaks = seq(0,1,0.25), labels = seq(0,1,0.25) * 100) + 
  scale_y_continuous(breaks = seq(0,1,0.25), labels = seq(0,1,0.25) * 100)


#Pole pod krzywa ROC
roc_curve
