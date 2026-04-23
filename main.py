import mne 
from mne.preprocessing import ICA
import numpy as np
from scipy.signal import welch
from mne.datasets import sample 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

data_path = sample.data_path()
raw = mne.io.read_raw_fif(data_path / 'MEG/sample/sample_audvis_raw.fif', preload=True)
print (raw.info)
raw.plot()

raw.filter (l_freq=1.0, h_freq=40.0) #φιλτραρει κατω απο 1Hz και πανω απο 40Hz
raw.notch_filter(freqs=50) #αφαιρει ταση απο πριζες
raw.set_eeg_reference('average', projection=True) #Επειδή η τάση μετριέται ως διαφορά δυναμικού, ορίζεις ένα "σημείο μηδέν" (εδώ τον μέσο όρο όλων των ηλεκτροδίων).
raw.apply_proj() #Εφαρμόζει τις προβολές που ορίστηκαν παραπάνω (π.χ. το μέσο όρο ως αναφορά) στα δεδομένα, ώστε να είναι έτοιμα για ανάλυση.
ica = ICA(n_components=20, random_state=42) # Η ICA είναι μια τεχνική που διαχωρίζει τα σήματα σε ανεξάρτητες πηγές, βοηθώντας να αφαιρεθούν οι αρνητικές επιρροές όπως οι κινήσεις των ματιών ή οι παλμοί της καρδιάς.
ica.fit(raw)
ica.plot_components() #ο φιλτράρισμα (Low-pass/High-pass) κόβει συχνότητες, ενώ το ICA κόβει "πηγές" θορύβου που τυχαίνει να έχουν τις ίδιες συχνότητες με τον εγκέφαλο.

events = mne.find_events (raw) #Δεν μας ενδιαφέρει όλη η καταγραφή, αλλά μόνο τι συνέβη ακριβώς τη στιγμή που ο ασθενής άκουσε έναν ήχο (auditory stimulus).
event_id = {"auditory/left": 1, 'auditory/right': 2} 

epochs = mne.Epochs(raw, events, event_id, #Είναι μικρά "παράθυρα" χρόνου (π.χ. από -0.2 έως 0.5 δευτερόλεπτα) γύρω από κάθε ήχο.
                    tmin=-0.2, tmax=0.5,
                    baseline=(-0.2, 0), #Πάρε το μέσο όρο της τάσης στα 200ms πριν το ερέθισμα (από το -0.2 έως το 0 δευτερόλεπτα) και αφαίρεσέ τον από όλο το epoch" ωστε η δραστηριότητα που βλέπεις μετά το ερέθισμα να ξεκινάει από το μηδέν
                    preload=True) #φορτώνει τα δεδομένα στη RAM για ταχύτερη επεξεργασία
epochs.plot()
print(epochs)


def band_power(epoch_data, sfreq, band): #Η συνάρτηση υπολογίζει την ισχύ σε μια συγκεκριμένη συχνότητα (π.χ. alpha 8-13Hz) για κάθε κανάλι και κάθε epoch.
    fmin, fmax = band #π.χ. για alpha, fmin=8 και fmax=13
    freqs, psd = welch(epoch_data, sfreq, nperseg=sfreq) #Αυτό σου δίνει ανάλυση 1 Hz (δηλαδή μπορείς να δεις ξεχωριστά το 8Hz από το 9Hz).
    idx = np.logical_and(freqs >= fmin, freqs <= fmax) #Βρίσκει τα indices των συχνοτήτων που βρίσκονται εντός του εύρους της μπάντας (π.χ. 8-13Hz για alpha).
    return np.mean(psd[:, idx], axis=1) #Το psd[:, idx] σημαίνει "πάρε όλα τα κανάλια, αλλά μόνο τις συχνότητες της μπάντας".

#Το axis=1 σημαίνει "βγάλε έναν μέσο όρο για κάθε κανάλι ξεχωριστά".

sfreq = epochs.info['sfreq'] # Αν sfreq = 600, σημαίνει ότι έχουμε 600 μετρήσεις ανά δευτερόλεπτο.
data = epochs.get_data() # Επιστρέφει ένα 3D array με διαστάσεις (n_epochs, n_channels, n_times) 
#Πόσες επαναλήψεις ερεθισμάτων έχουμε (π.χ. 100 ήχοι).
#Πόσα ηλεκτρόδια έχουμε στο κεφάλι (π.χ. 64 κανάλια).
#Τα σημεία δεδομένων σε κάθε epoch (π.χ. αν το epoch είναι 1sec και το sfreq 600, τότε είναι 600 σημεία).

bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
}

#Αν έχεις 64 ηλεκτρόδια (κανάλια), η band_power θα σου επιστρέψει 64 νούμερα. Ένα νούμερο για κάθε σημείο του εγκεφάλου, που αντιπροσωπεύει την ένταση της συγκεκριμένης μπάντας (π.χ. Alpha) σε εκείνο το σημείο.

features = []
for epoch in data: #Για ΚΑΘΕ epoch, υπολογίζεις 4 μπάντες Κάθε μπάντα σου δίνει 306 νούμερα (αν έχεις 306 κανάλια)
    row = np.concatenate([band_power(epoch, sfreq, b) for b in bands.values()]) #Το concatenate ενώνει αυτά τα 4 groups σε μια μεγάλη "γραμμή
    features.append(row) #Άρα κάθε row έχει 4 * 306 = 1224 στοιχεία (Features)!

X = np.array(features)  #φτιάχνεις έναν πίνακα με διαστάσεις (Αριθμός_Epochs, 1224)
y = epochs.events[:, 2]  # παίρνει την τρίτη στήλη του πίνακα των events. Αυτή η στήλη περιέχει τα IDs
epochs.plot_psd(fmin=1, fmax=40) #Ένα γράφημα όπου ο οριζόντιος άξονας (x) είναι οι συχνότητες (Hz) και ο κάθετος (y) είναι η δύναμη (dB).
#Στο MNE, ο πίνακας events έχει πάντα 3 στήλες:

#Στήλη 0: Το δείγμα (sample index) που συνέβη το γεγονός.
#Στήλη 1: Τιμή πριν το event (συνήθως 0).
#Στήλη 2: Το ID του event (π.χ. 1="auditory/left"). Αυτό είναι που θέλει να προβλέψει το Machine Learning!

epochs.plot_psd_topomap(bands={'alpha': (8, 13)}) #Ένα topomap είναι ένας χάρτης του κεφαλιού που δείχνει την κατανομή της ισχύος σε μια συγκεκριμένη μπάντα (π.χ. Alpha) σε όλα τα κανάλια. 

evoked = epochs['auditory/left'].average() #Παίρνει όλα τα epochs (τις επαναλήψεις) όπου ο χρήστης άκουσε ήχο από αριστερά και βγάζει τον μέσο όρο τους. Το EEG είναι γεμάτο τυχαίο θόρυβο. Αν όμως τα προσθέσεις όλα μαζί, ο θόρυβος (που είναι τυχαίος) αλληλοεξουδετερώνεται και μένει μόνο η σταθερή αντίδραση του εγκεφάλου στον ήχο. Αυτό το μέσο σήμα ονομάζεται "evoked response" ή "ERP" (Event-Related Potential).
evoked.plot()
evoked.plot_topomap(times=[0.1, 0.2, 0.3])

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis()),
])

scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
print(f"Cross-val accuracy: {scores.mean():.2f}")