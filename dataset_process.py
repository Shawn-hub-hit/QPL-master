
from email import header
from Config import *
os.chdir(sys.path[0])
import re


class Interactions(object):
    def __init__(self, parser, log_):
        self._parser = parser
        self.log_ = log_
        self.test_relation_df = pd.read_csv(self._parser.test_file, header=0)
        self.test_relation_df['record_id'] = list(range(len(self.test_relation_df)))

        self.left_text_df = pd.read_csv(self._parser.left_text_file, header=0)
        self.left_text_df = self.left_text_df[['id_left', 'location_left', 'text_left']]

        self.query_maxlen = self._parser.fix_left_length
        self.doc_maxlen = self._parser.fix_right_length
        self.right_text_df = pd.read_csv(self._parser.right_text_file, header=0)
        self.right_text_df = self.right_text_df[['id_right', 'location_right', 'text_right']]
        
        self.log_.info(f'{self.test_relation_df.shape}, {self.left_text_df.shape}, {self.right_text_df.shape}')
        self.left_text_df.set_index('id_left', inplace=True)
        self.right_text_df.set_index('id_right', inplace=True)
        self.change_type(self.test_relation_df)
    
        self.data_process()
        
        self.left_text_df['location_left'] = self.left_text_df['location_left'].map(eval)
        self.right_text_df['location_right'] = self.right_text_df['location_right'].map(eval)
        self.left_text_df['text_left'] = self.left_text_df['text_left'].map(eval)
        self.right_text_df['text_right'] = self.right_text_df['text_right'].map(eval)
        
        poi_coodis = self.right_text_df['location_right'].tolist()
        
        self.poi_coodis = np.array(poi_coodis)
        self.poi_num = len(self.right_text_df)

        self.left_text_df['location_left_l'] = self.left_text_df['location_left'].map(self.location_map2)

        self.right_text_df['location_right_l'] = self.right_text_df['location_right'].map(self.location_map2)

    def generate_negs(self, id_left):
        query_text = self.left_text_df.loc[id_left]['text_left']
        location_query = self.left_text_df.loc[id_left]['location_left']
        samples_t_idx, samples_l_idx = self.sampling.search_rows(query_text, location_query)
        candidate_pois = list(samples_t_idx) + list(samples_l_idx)
        return candidate_pois

    def change_type(self, relation_df):
        relation_df['label'] = relation_df['label'].astype(float)
        relation_df['distance'] = relation_df['distance'].astype(float)
        relation_df['id_left'] = relation_df['id_left'].astype(int)
        relation_df['id_right'] = relation_df['id_right'].astype(int)
        relation_df['userId'] = relation_df['userId'].astype(int)

    def data_process(self):

        self.embedding_matrix = self.load_chinese_embedding()

    def load_chinese_embedding(self):

        with open(self._parser.emb_matrix_file, 'rb') as file:
            matrix = pickle.load(file) 

        return matrix


    def pack_data(self, index, relation, _stage, with_distance):

        index = self._convert_to_list_index(index, len(relation))
        relation = relation.iloc[index].reset_index(drop=True)
        relation_l = []
        for index, row in relation.iterrows():
            record_id = row.record_id
            user_id = row.userId
            id_left = row.id_left
            id_right = row.id_right
            # query_text = self.left_text_df.loc[id_left]['text_left']
            location_query = self.left_text_df.loc[id_left]['location_left']
            distance = row.distance

            candidates_distance = spatial.distance.cdist(np.array([location_query]), self.poi_coodis)[0]/1000
            max_score_spatial = np.max(candidates_distance)

            if _stage == 'test':
                candidates_distance = list(1 - candidates_distance / max_score_spatial)  # convert to distance score
                candidate_pois = list(range(self.poi_num))
                idx_ = candidate_pois.index(id_right)
                del candidate_pois[idx_]
                del candidates_distance[idx_]

            relation_l.append([record_id, user_id, id_left, 1.0, id_right, 1-distance/max_score_spatial])

            # tile the negative samples
            neg_list = [[record_id, user_id, id_left, 0.0]]
            neg_list = neg_list * len(candidate_pois)
            df_neg_l = pd.DataFrame(neg_list, columns=['record_id', 'userId','id_left', 'label'])
            df_neg_l['id_right'] = candidate_pois
            if with_distance:
                df_neg_l['distance'] = candidates_distance
            neg_list_ = df_neg_l.values.tolist()
            relation_l = relation_l + neg_list_
        
        relation2 = pd.DataFrame(relation_l, columns=['record_id', 'userId', 'id_left', 'label', 'id_right', 'distance'])

        left = self.left_text_df.loc[relation2['id_left'].unique()]
        right = self.right_text_df.loc[relation2['id_right'].unique()]

        return relation2, left, right

    def unpack(self, data_pack):
        relation, left, right = data_pack
        index = list(range(len(relation)))
        left_df = left.loc[relation['id_left'][index]].reset_index()
        right_df = right.loc[relation['id_right'][index]].reset_index()
        joined_table = left_df.join(right_df)

        for column in relation.columns:
            if column not in ['id_left', 'id_right']:
                labels = relation[column][index].to_frame()
                labels = labels.reset_index(drop=True)
                joined_table = joined_table.join(labels)

        columns = list(joined_table.columns)
        y = np.vstack(np.asarray(joined_table['label']))
        x = joined_table[columns].to_dict(orient='list')
        for key, val in x.items():
            x[key] = np.array(val)

        return x, y


    def location_map2(self, obj):
        if dataname == 'Beijing_data':
            dXMin = 366950.2449227313       #39.53912
            dXMax = 541972.8942822593        #40.96375
            dYMin = 4377750.661527712        #115.4517
            dYMax = 4534852.758361747        #117.4988
        elif dataname == 'Shanghai_data':
            dXMin = 211537      
            dXMax = 401730      
            dYMin = 3321378       
            dYMax = 3543503      

        self.GridCount_x = int((dXMax - dXMin) * 1.0 / self._parser.grid_size) + 2
        self.GridCount_y = int((dYMax - dYMin) * 1.0 / self._parser.grid_size) + 2
        
        nXCol = int((obj[0] - dXMin) / self._parser.grid_size)
        nYCol = int((obj[1] - dYMin) / self._parser.grid_size)

        return [nXCol, nYCol]

    def _convert_to_list_index(self, index, length):
        if isinstance(index, int):
            index = [index]
        elif isinstance(index, slice):
            index = list(range(*index.indices(length)))
        return index


class my_Datasets(data.Dataset):

    def __init__(self, relation, interaction, stage, batch_size=32, resample=False, shuffle=True):
        self._orig_relation = relation.copy()
        self.interaction = interaction
        self._batch_size = batch_size
        self._batch_indices = None
        self._shuffle = shuffle
        self.reset_index()
        self._resample = resample
        self.stage = stage
        self._with_distance = True

    def reset_index(self):
        index_pool = []
        step_size = 1
        num_instances = int(len(self._orig_relation) / step_size)
        for i in range(num_instances):
            lower = i * step_size
            upper = (i+1) * step_size
            indices = list(range(lower, upper))
            if indices:
                index_pool.append(indices)
        if self._shuffle == True:
            np.random.shuffle(index_pool)
        self._batch_indices = []
        for i in range(math.ceil(num_instances / self._batch_size)):
            lower = self._batch_size * i
            upper = self._batch_size * (i + 1)
            candidates = index_pool[lower:upper]
            candidates = sum(candidates, [])
            self._batch_indices.append(candidates)

    def __getitem__(self, item):
        if isinstance(item, slice):
            indices = sum(self._batch_indices[item], [])
        elif isinstance(item, Iterable):
            indices = [self._batch_indices[i] for i in item]
        else:
            indices = self._batch_indices[item]
        data_dict = self.interaction.pack_data(indices, self._orig_relation, self.stage, with_distance=self._with_distance)
        x, y = self.interaction.unpack(data_dict)

        return x, y

    def __iter__(self):
        """Create a generator that iterate over the Batches."""
        if self._resample or self._shuffle:
            self.on_epoch_end()
        for i in range(len(self)):
            yield self[i]

    def on_epoch_end(self):
        """Reorganize the index array if needed."""
        self.reset_index()

    def __len__(self):
        """Get the total number of batches."""
        return len(self._batch_indices)
