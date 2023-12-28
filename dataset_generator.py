import pandas as pd


## Путь до файлов датасета
path_to_file_1 = 'data/ga_sessions.csv'
path_to_file_2 = 'data/ga_hits.csv'


def generate_dataset():
    def create_dataset_key_action(df_sessions, df_hits):

        def target_action(df):
            if df['event_action'] == 'sub_car_claim_click':
                return 1
            elif df['event_action'] == 'sub_car_claim_submit_click':
                return 1
            elif df['event_action'] == 'sub_open_dialog_click':
                return 1
            elif df['event_action'] == 'sub_custom_question_submit_click':
                return 1
            elif df['event_action'] == 'sub_call_number_click':
                return 1
            elif df['event_action'] == 'sub_callback_submit_click':
                return 1
            elif df['event_action'] == 'sub_submit_success':
                return 1
            elif df['event_action'] == 'sub_car_request_submit_click':
                return 1
            else:
                return 0

        df_hits['y'] = df_hits.apply(target_action,
                                     axis=1)  # отмечаем на датасете действия, в которых клиенты совершили target
        df_1 = df_hits[df_hits['y'] == 1].copy()  # отделим действия target от остальных
        df_0 = df_hits[df_hits['y'] == 0].copy()
        df_1 = df_1.sort_values(by=['hit_date', 'hit_time'])
        df_0 = df_0.sort_values(by=['hit_date', 'hit_time'])

        # удалим дубликаты сессий target, так как если клиент совершил несколько target, то они приравниваются к одному target
        # оставим только первое ключевое действие в df_1, а в df_0 последнее действие

        df_1 = df_1.drop_duplicates(subset=['session_id'], keep='first')
        df_0 = df_0.drop_duplicates(subset=['session_id'], keep='last')
        df_1 = df_1.set_index('session_id')
        df_0 = df_0.set_index('session_id')
        df_action = pd.concat((df_1, df_0), axis=0, ignore_index=False)

        # уникальных индексов меньше, значит присутствуют дубликаты. Удалим дубликаты с сохранением первых,
        # так как y==1 стоит вначале датафрейма они сохранятся

        df_action = df_action.reset_index().drop_duplicates(subset='session_id', keep='first').set_index('session_id')
        df_action = df_action[['hit_number', 'hit_page_path', 'hit_date', 'hit_time', 'y']]
        df_sessions = df_sessions.set_index('session_id')
        df_sessions['hit_number'] = df_action['hit_number']
        df_sessions['hit_page_path'] = df_action['hit_page_path']
        df_sessions['hit_date'] = df_action['hit_date']
        df_sessions['hit_time'] = df_action['hit_time']
        df_sessions['y'] = df_action['y']  # колонки df_action объединяются с df_sessions по индексу

        # разделим df_sessions датасет на два датасета - сессии где клиент совершил целевое target и не совершил

        df_sessions_1 = df_sessions[df_sessions.y == 1].copy()
        df_sessions_0 = df_sessions[df_sessions.y == 0].copy()

        # удалим дубликаты сессий в обоих датасетах по client_id

        df_sessions_1 = df_sessions_1.drop_duplicates(subset=['client_id'], keep='first')
        df_sessions_0 = df_sessions_0.drop_duplicates(subset=['client_id'], keep='last')
        df_sessions_y = df_sessions_1.append(df_sessions_0).copy()  # объединим датасеты, вверху будут target = 1

        # теперь клиенты target = 1 стоят в начате датафрейма и при удалении дубликатов сессий они сохранятся (keep='first')
        # в df_sessions_y сохранились дубликаты так как в target=0 сохранились client_id да того как они совершили target = 1

        df_sessions_y = df_sessions_y.drop_duplicates(subset=['client_id'], keep='first')

        # выберем колонки, которые будут участвовать в модели

        df_sessions_y = df_sessions_y.set_index('client_id')
        df_key_action = df_sessions_y[[
            'visit_time', 'visit_number', 'hit_number', 'hit_page_path', 'hit_date', 'hit_time',
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
            'device_category', 'device_os', 'device_brand', 'device_model', 'device_screen_resolution',
            'device_browser', 'geo_country', 'geo_city', 'y'
        ]].copy()

        df_key_action = df_key_action.sample(frac=1)  # перемешать.

        return df_key_action.copy()

    df_1 = pd.read_csv(path_to_file_1)
    df_2 = pd.read_csv(path_to_file_2)
    df_result = create_dataset_key_action(df_1, df_2)

    return df_result.to_csv('data/df_key_action.csv')


if __name__ == '__main__':
    generate_dataset()