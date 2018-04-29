import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net


class ComboModel(object):
    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.c, self.q, self.ch, self.qh, self.qa_id, self.bad_y1, self.bad_y2, self.y1, self.y2 = batch.get_next()
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False)
        self.char_mat = tf.get_variable(
            "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))

        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        if opt:
            N, CL = config.batch_size, config.char_limit
            self.c_maxlen = tf.reduce_max(self.c_len)
            self.q_maxlen = tf.reduce_max(self.q_len)
            self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
            self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
            self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
            self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
            self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
            self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
            self.bad_y1 = tf.slice(self.bad_y1, [0, 0], [N, self.c_maxlen])
            self.bad_y2 = tf.slice(self.bad_y2, [0, 0], [N, self.c_maxlen])
            self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
            self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
        else:
            self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

        self.ch_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
        self.qh_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

        self.ready()

        if trainable:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            #self.opt = tf.train.AdadeltaOptimizer(
            #    learning_rate=self.lr, epsilon=1e-6)
            self.opt = tf.train.AdamOptimizer()
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

    def ready(self):
        config = self.config
        N, PL, QL, CL, d, dc, dg = config.batch_size, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.char_hidden
        gru = cudnn_gru if config.use_cudnn else native_gru

        with tf.variable_scope("emb"):
            with tf.variable_scope("char"):
                ch_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.ch), [N * PL, CL, dc])
                qh_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.qh), [N * QL, CL, dc])
                ch_emb = dropout(
                    ch_emb, keep_prob=config.keep_prob, is_train=self.is_train)
                qh_emb = dropout(
                    qh_emb, keep_prob=config.keep_prob, is_train=self.is_train)
                cell_fw = tf.contrib.rnn.GRUCell(dg)
                cell_bw = tf.contrib.rnn.GRUCell(dg)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, ch_emb, self.ch_len, dtype=tf.float32)
                ch_emb = tf.concat([state_fw, state_bw], axis=1)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, qh_emb, self.qh_len, dtype=tf.float32)
                qh_emb = tf.concat([state_fw, state_bw], axis=1)
                qh_emb = tf.reshape(qh_emb, [N, QL, 2 * dg])
                ch_emb = tf.reshape(ch_emb, [N, PL, 2 * dg])

            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)

            self.c_emb = c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

            bad_c_emb = tf.stop_gradient(c_emb)
            bad_q_emb = tf.stop_gradient(q_emb)

        with tf.variable_scope("encoding"):
            rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=bad_c_emb.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            self.c_rnn = rnn(bad_c_emb, seq_len=self.c_len)
            self.q_rnn = rnn(bad_q_emb, seq_len=self.q_len)

            badptr_c = tf.stop_gradient(self.c_rnn)
            badptr_q = tf.stop_gradient(self.q_rnn)
            old_rnn = rnn

        with tf.variable_scope("badptr_attention"):
            qc_att, self.badptr_qc_att = dot_attention(badptr_c, badptr_q, mask=self.q_mask, hidden=d,
                                   keep_prob=config.keep_prob, is_train=self.is_train, give=True)
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            
            self.att = [rnn(qc_att, seq_len=self.c_len)]
            self.att += [self.att[-1][:,-1,:]]
        
        with tf.variable_scope("badptr_dense"):
            for _ in range(3):
                self.att += [tf.nn.dropout(tf.keras.layers.Dense(300)(self.att[-1]), keep_prob=config.keep_prob)]

        with tf.variable_scope("badptr"):
            init = self.att[-1]
            pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
            )[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            logits1, logits2 = pointer(init, self.att[0], d, self.c_mask)

        with tf.variable_scope("badptr_predict"):
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, 15)
            self.bad_yp1_distrib = tf.reduce_max(outer, axis=2)
            self.bad_yp2_distrib = tf.reduce_max(outer, axis=1)
            self.bad_yp1 = tf.argmax(self.bad_yp1_distrib, axis=1)
            self.bad_yp2 = tf.argmax(self.bad_yp2_distrib, axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits1, labels=tf.stop_gradient(self.bad_y1))
            losses2 = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits2, labels=tf.stop_gradient(self.bad_y2))
            self.loss = tf.reduce_mean(losses + losses2)

        # recompute c with bitmask
        left = tf.sequence_mask(self.bad_yp1, tf.shape(c_emb)[1])
        right = tf.logical_not(tf.sequence_mask(self.bad_yp2+1, tf.shape(c_emb)[1]))
        self.combo = combo = tf.logical_or(left, right)

        ### FOR TESTING ###
        ## self.combo = combo = tf.cast(tf.ones_like(combo), tf.bool)

        def adjust(c_emb_combo):
            c_emb, combo = c_emb_combo
            foo = c_emb
            bar = tf.boolean_mask(foo, combo)

            return tf.cond(tf.logical_and(tf.equal(combo[0], False), tf.equal(combo[1], True)),
                           false_fn=lambda: tf.pad(bar, [[0, tf.shape(foo)[0] - tf.shape(bar)[0]], [0, 0]]),
                           true_fn=lambda: foo)

        self.c_emb_new = c_emb_new = tf.map_fn(adjust, (c_emb, combo), dtype=(tf.float32))
        self.c_len = tf.reduce_sum(tf.cast(tf.logical_and(self.c_mask, self.combo), tf.int32), axis=-1)
        self.c_mask = tf.sequence_mask(tf.reduce_sum(tf.cast(tf.logical_and(self.c_mask, self.combo), tf.int32), axis=-1), tf.shape(self.c_mask)[1])

        with tf.variable_scope("encoding", reuse=True):
            rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=c_emb.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train, super_hacky_reload=True)
            #### SEQ LEN HAS TO BE FIXED!!!! ####
            c = rnn(c_emb_new, seq_len=self.c_len)
            q = rnn(q_emb, seq_len=self.q_len)

        self.c_ck = c
        self.q_ck = c

        ### MAKE SURE THESE ARE RUN!!! ###
        print('RUN ASSIGN TRICK OPS (model.assign_trick_ops)!!')
        self.assign_trick_ops = []
        for i in range(len(rnn.init_fw)):
            self.assign_trick_ops += [tf.assign(rnn.init_fw[i], old_rnn.init_fw[i])]
            self.assign_trick_ops += [tf.assign(rnn.init_bw[i], old_rnn.init_bw[i])]

        with tf.variable_scope("attention"):
            qc_att, self.qc_att = dot_attention(c, q, mask=self.q_mask, hidden=d,
                                   keep_prob=config.keep_prob, is_train=self.is_train, give=True)
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            att = rnn(qc_att, seq_len=self.c_len)

        self.att_ck = att

        with tf.variable_scope("match"):
            self_att = dot_attention(
                att, att, mask=self.c_mask, hidden=d, keep_prob=config.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=self_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            match = rnn(self_att, seq_len=self.c_len)

        self.match_ck = match

        with tf.variable_scope("pointer"):
            init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
                        keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
            )[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            logits1, logits2 = pointer(init, match, d, self.c_mask)

        with tf.variable_scope("predict"):
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, 15)
            self.yp1_distrib = tf.reduce_max(outer, axis=2)
            self.yp2_distrib = tf.reduce_max(outer, axis=1)
            self.yp1 = tf.argmax(self.yp1_distrib, axis=1)
            self.yp2 = tf.argmax(self.yp2_distrib, axis=1)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
