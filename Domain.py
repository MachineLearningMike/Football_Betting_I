#import tensorflow_datasets as tfds
import tensorflow as tf



class Domain() :
    def __init__(self, bufferSize, batchSize, max_length, source, target):

        self.max_length = max_length

        sourceOOVToken = 1
        self.tokenizer_source = tf.keras.preprocessing.text.Tokenizer( filters='!"#$%&()*+,-./:;?@[\\]^`{|}~\t\n', lower = True, split=' ', \
            char_level=False, oov_token = sourceOOVToken, document_count=0 )
        self.tokenizer_source.fit_on_texts( source )

        self.sourceOOVToken = sourceOOVToken
        self.sourceStartToken = len(self.tokenizer_source.word_counts) + 2
        self.sourceEndToken = len(self.tokenizer_source.word_counts) + 3

        targetOOVToken = 1
        self.tokenizer_target = tf.keras.preprocessing.text.Tokenizer( filters='!"#$%&()*+,-./:;?@[\\]^`{|}~\t\n', lower = True, split=' ', \
            char_level=False, oov_token = targetOOVToken, document_count=0 )
        self.tokenizer_target.fit_on_texts( target )

        self.targetOOVToken = targetOOVToken
        self.targetStartToken = len(self.tokenizer_target.word_counts) + 2
        self.targetEndToken = len(self.tokenizer_target.word_counts) + 3

        nExamples = source.shape[-1]
        assert target.shape[-1] == nExamples
        nTrainExamples = int(nExamples * 0.75)  # 0.75
        train_source = source[ : nTrainExamples ]
        val_source = source[ nTrainExamples : ]
        train_target = target[ : nTrainExamples ]
        val_target = target[ nTrainExamples : ]

        train_dataset = tf.data.Dataset.from_tensor_slices( ( train_source, train_target ) )
        val_dataset = tf.data.Dataset.from_tensor_slices( ( val_source, val_target ) )

        #self.tokenizer_target = tfds.features.text.SubwordTextEncoder.build_from_corpus((en.numpy() for pt, en in train_examples), target_vocab_size=2**13)
        #self.tokenizer_source = tfds.features.text.SubwordTextEncoder.build_from_corpus((pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)

        train_dataset = train_dataset.map(self.__tf_encode__)
        train_dataset = train_dataset.filter(self.__filter_max_length__)

        # cache the dataset to memory to get a speedup while reading from it.
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(bufferSize).padded_batch(batchSize, padded_shapes= ([None], [None]))
        self.train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = val_dataset.map(self.__tf_encode__)
        self.val_dataset = val_dataset.filter(self.__filter_max_length__).padded_batch(batchSize, padded_shapes= ([None], [None]))
      
    def __encode__(self, lang1, lang2):
        lang1 = [self.sourceStartToken] + self.tokenizer_source.texts_to_sequences( [lang1.numpy().decode()] )[0] + [self.sourceEndToken]
        lang2 = [self.targetStartToken] + self.tokenizer_target.texts_to_sequences( [lang2.numpy().decode()] )[0] + [self.targetEndToken]

        return lang1, lang2

    def __tf_encode__(self, pt, en):
        result_pt, result_en = tf.py_function(self.__encode__, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en

    def __filter_max_length__(self, x, y):
        return tf.logical_and(tf.size(x) <= self.max_length, tf.size(y) <= self.max_length)
